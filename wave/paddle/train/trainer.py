#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import itertools
import six
import inspect
from contextlib import contextmanager
from six.moves import zip, map
import logging
from time import time

import paddle.fluid as F
import paddle.fluid.layers as L

from wave.types import RunMode, StopException, SummaryRecord, StopException, ModelSpec, InferenceSpec
from wave.paddle import summary, collection
from wave.paddle.data.functional import Dataset
from wave.paddle.train import distribution
from wave.train.model import Model
from wave.paddle.train.monitored_executor import Saver
from wave.paddle.train import hooks, metrics

from wave.paddle.train.monitored_executor import MonitoredExecutor

log = logging.getLogger(__name__)

__all__ = ['train_and_eval', 'predict']


def get_parallel_exe(program, loss, dev_count):
    exec_strategy = F.ExecutionStrategy()
    exec_strategy.num_threads = 4  #2 for fp32 4 for fp16
    exec_strategy.use_experimental_executor = True
    exec_strategy.num_iteration_per_drop_scope = 10  #important shit

    build_strategy = F.BuildStrategy()
    build_strategy.remove_unnecessary_lock = False
    #build_strategy.fuse_broadcast_ops = True

    log.info('replica id %d of %d' % (distribution.status.replica_id,
                                      distribution.status.num_replica))
    train_exe = F.ParallelExecutor(
        use_cuda=True,
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy,
        main_program=program,
        num_trainers=distribution.status.num_replica,
        trainer_id=distribution.status.replica_id)
    return train_exe


def build_net(model_fn_or_model, features, mode, params, run_config):
    if issubclass(model_fn_or_model, Model):

        def model_fn(features, mode, params, run_config):
            if mode != RunMode.PREDICT:
                fea, label = features[:-1], features[-1]
            else:
                fea = features

            model = model_fn_or_model(params, mode, run_config=run_config)
            pred = model.forward(fea)

            if mode == RunMode.TRAIN:
                loss = model.loss(pred, label)
                model.backward(loss)
                return ModelSpec(loss=loss, predictions=pred, mode=mode)
            elif mode == RunMode.EVAL:
                loss = model.loss(pred, label)
                me = model.metrics(pred, label)

                pred_list = pred if isinstance(pred, (list, tuple)) else [pred]
                inf_spec = InferenceSpec(inputs=fea, outputs=pred_list)
                if 'loss' not in me:
                    me['loss'] = metrics.Mean(loss)
                return ModelSpec(
                    loss=loss,
                    predictions=pred,
                    metrics=me,
                    mode=mode,
                    inference_spec=inf_spec)
            elif mode == RunMode.PREDICT:
                return ModelSpec(predictions=pred, mode=mode)
            else:
                raise RuntimeError('unknown run mode %s' % mode)
    elif inspect.isfunction(model_fn_or_model):
        model_fn = model_fn_or_model
    else:
        raise ValueError('unknown model %s' % model_fn_or_model)

    model_spec = model_fn(
        features=features, mode=mode, params=params, run_config=run_config)
    if mode == RunMode.TRAIN:
        assert model_spec.loss is not None
    elif mode == RunMode.EVAL:
        assert model_spec.metrics is not None
    elif mode == RunMode.PREDICT:
        assert model_spec.predictions is not None
    else:
        raise ValueError('unkonw mode %s' % mode)
    return model_spec


def predict(_shit=None,
            model_class_or_model_fn=None,
            params=None,
            model_dir=None,
            infer_dataset=None,
            run_config=None,
            steps=-1,
            split_batch=True):
    '''
    Perform predictoin
    will call `model_fn` twice and initiate user-specifed model in `wave.RunMode.TRAIN` mode and `wave.RunMode.EVAL` mode

    Args:
        model_class_or_model_fn(callable|wave.train.Model): `model_class_or_model_fn` be specified in 2 ways:
            1. subclass of wave.train.Model which implements:
                1. \_\_init\_\_       (hyper_param, mode, run_config)
                2. forward            (features) => (prediction)
                
            2. a model_fn takes following args:
                1. features
                2. param
                3. mode
                4. run_config(optional)
               and returns a `wave.ModelSpec` 
        params: any python object, will pass to your `model_fn` or `wave.train.Model`
        model_dir (str):  path to your model save directory.
        infer_dataset (wave.data.Dataset): should not `shuffle` or `repeat`
        run_config (wave.RunConfig): will pass to your  `model_fn` or `wave.train.Model`
        steps (int): steps to predict, if -1 is specifed, will stop when `StopException` is raised in `infer_dataset`
        split_batch (bool): if True, prediction of each example in a batch is returned.

    Yields:
        Evaluated values of predictions tensors.

    '''
    if _shit is not None:
        raise ValueError('specify keyword args to this function')
    if model_class_or_model_fn is None or params is None or model_dir is None or infer_dataset is None:
        raise ValueError(
            'some argument is None: model_class_or_model_fn:%s params:%s run_config:%s train_dataset:%s'
            % (model_class_or_model_fn, params, run_config, train_dataset))

    if not os.path.exists(model_dir):
        raise ValueError('model dir not found %s' % model_dir)

    program = F.Program()
    startup_prog = F.Program()
    with F.program_guard(program, startup_prog):
        with F.unique_name.guard():
            fea = infer_dataset.features()
            log.info('Building Predict Graph...')
            model_spec = build_net(model_class_or_model_fn, fea,
                                   RunMode.PREDICT, params, run_config)
            log.info('Building Predict Graph: Done')
            log.info('Memory optimizing...')
            F.memory_optimize(input_program=program)
            log.info('Memory optimizing: Done')
    program = program.clone(for_test=True)
    start_exe = F.Executor(F.CUDAPlace(0))
    start_exe.run(startup_prog)

    F.io.load_vars(
        start_exe,
        model_dir,
        main_program=program,
        predicate=F.io.is_persistable)

    pred = model_spec.predictions
    pred_list = pred if isinstance(pred, (list, tuple)) else [pred]

    dev_list = F.cuda_places()  #list all visible divices
    if len(dev_list) > 1:
        log.warm(
            'Executing multi card prediction, No. of cards: %d > 1. will drop remainder'
            % len(dev_list))
    predict_exe = get_parallel_exe(program, model_spec.predictions,
                                   len(dev_list))
    try:
        log.info('Runining predict from dir: %s' % model_dir)
        for data in infer_dataset.start(dev_list):
            res = predict_exe.run(fetch_list=pred_list, feed=data)
            if split_batch:
                res = map(lambda i: i.tolist(), res)
                res = zip(*res)  # transpose
                for r in res:
                    yield r
            else:
                yield res
    except F.core.EOFException:
        log.debug('Predict done')


def train_and_eval(_shit=None,
                   model_class_or_model_fn=None,
                   params=None,
                   run_config=None,
                   train_dataset=None,
                   eval_dataset=None,
                   warm_start_setting=None,
                   train_hooks=[],
                   eval_hooks=[],
                   exporters=[]):
    '''
    Perform train and evaluate procesure. 
    will call `model_fn` and initiate user-specifed model in `wave.RunMode.PREDICT` mode 

    Args:
        model_class_or_model_fn(callable|wave.train.Model): `model_class_or_model_fn` be specified in 2 ways:
            1. subclass of wave.train.Model which implements:
                1. \_\_init\_\_       (hyper_param, mode, run_config)
                2. forward            (features) => (prediction)
                3. backword           (loss) => None
                4. loss               (predictoin) => (loss)
                5. metrics (optional) (prediction) => (dict of wave.Metrics)
                
            2. a model_fn takes following args:
                1. features
                2. param
                3. mode
                4. run_config(optional)
               and returns a `wave.ModelSpec`

        params: any python object, will pass to your `model_fn` or `wave.train.Model`
        run_config (wave.RunConfig): run_config.max_steps should not be None.
        train_dataset (wave.paddle.data.Dataset): training will stop if global_step > run_config.max_steps.
        eval_dataset (wave.paddle.data.Dataset|dict): Optional, if Dict of wave.data.Dataset were specified, will perform evluatation on every evaluation sets and report results.
        warm_start_setting (wave.WarmStartSetting): Optional. warm start variable will overwrite model variable.
        train_hooks (list of wave.paddle.train.RunHook): Optional.
        eval_hooks (list of wave.paddle.train.RunHook): Optional.
        exporters (list of wave.paddle.train.Exporter): Optional.
    '''
    if _shit is not None:
        raise ValueError('specify keyword args to this function')
    if model_class_or_model_fn is None or params is None or run_config is None or train_dataset is None:
        raise ValueError(
            'some argument is None: model_class_or_model_fn:%s params:%s run_config:%s train_dataset:%s'
            % (model_class_or_model_fn, params, run_config, train_dataset))
    train_dataset.name = 'train'
    train_program = F.Program()
    startup_prog = F.Program()
    with F.program_guard(train_program, startup_prog):
        with F.unique_name.guard():
            with collection.Collections() as collections:
                log.info('Building Train Graph...')
                fea = train_dataset.features()
                model_spec = build_net(model_class_or_model_fn, fea,
                                       RunMode.TRAIN, params, run_config)
                log.info('Building Train Graph: Done')

            scalars = collections.get(collection.Key.SUMMARY_SCALAR)
            histograms = collections.get(collection.Key.SUMMARY_HISTOGRAM)
            skip_optimize_ops = collections.get(collection.Key.SKIP_OPTIMIZE)
            skip_opt = set()
            if skip_optimize_ops is not None:
                skip_opt |= set(skip_optimize_ops)
            if scalars is not None:
                skip_opt |= {t for _, t in scalars}
            if histograms is not None:
                skip_opt |= {t for _, t in histograms}
            skip_opt = list(skip_opt)
            log.info('skip memory optimize for %d ops' % len(skip_opt))
            log.info('Memory optimizing...')
            F.memory_optimize(
                input_program=train_program, skip_opt_set=skip_opt)
            log.info('Memory optimizing: Done')

    log.info(
        'Train with: \n> Run_config: %s\n> Params: %s\n> Train_model_spec: %s\n'
        % (repr(run_config), repr(params), repr(model_spec)))

    #init distribution env if envvir ATARASHI_DISCONFIG is set
    distribution.init_distribuition_env(train_program, startup_prog)

    if eval_dataset is not None:
        if not isinstance(eval_dataset, (dict, Dataset)):
            raise ValueError(
                'Eval dataset should be wave.Dataset of a list of that, got: %s'
                % eval_dataset)
        if isinstance(eval_dataset, Dataset):
            eval_dataset = {'eval': eval_dataset}
        ds_list = list(eval_dataset.values())
        for ds in ds_list:
            ds.name = 'eval'
        first = ds_list[0]
        for d in ds_list[1:]:
            if not first.__eq__(d):
                raise ValueError(
                    'eval dataset has different output_shapes or types: %s' %
                    repr(ds_list))
        eval_program = {}
        for name, ds in six.iteritems(eval_dataset):
            program = F.Program()
            with F.program_guard(program, startup_prog):
                #share var with Train net
                with F.unique_name.guard():
                    log.info('Building Eval Graph')
                    fea = ds.features()
                    eval_model_spec = build_net(model_class_or_model_fn, fea,
                                                RunMode.EVAL, params,
                                                run_config)
                    log.info('Done')
            program = program.clone(for_test=True)
            eval_program[name] = program

        def evaluate(name, train_state, ds, writer):
            try:  #[try -> with -> while]
                program = eval_program[name]
                eval_hook = hooks.EvalHook(
                    name,
                    eval_model_spec.metrics,
                    summary_writer=writer,
                )  #summary_writer is defined below...
                eval_hook.set_train_state(train_state)
                eval_run_hooks = [eval_hook]
                if run_config.eval_max_steps is not None:
                    eval_run_hooks.append(
                        hooks.StopAtStepHook(
                            run_config.eval_max_steps,
                            run_config.eval_max_steps,
                            msg='evaluating %s' % name))
                eval_run_hooks.extend(eval_hooks)
                eval_ds = ds.start(places=[single_card_place])
                with MonitoredExecutor(
                        start_exe,
                        program=program,
                        run_config=None,
                        run_hooks=eval_run_hooks, ) as eval_exe:
                    for data in eval_ds:
                        eval_exe.run(feed=data)
            except (F.core.EOFException, StopException):
                pass
            return eval_hook.result

        log.info('Eval with: \n> Eval_model_spec %s' % repr(eval_model_spec))

    dev_list = F.cuda_places()  #list all visible divices
    log.info('Visible device %s' % repr(dev_list))
    #dev_list = [int(i) for i in os.environ.get('FLAGS_selected_gpus').split(',')]
    #log.debug('GPU list is specified %s' % repr(dev_list))
    #dev_count = len(dev_list)

    #param broadcast happened when creating ParallelProgram, init before this

    #The order of this 3 steps really matters
    #1. init train
    #single_card_place = F.CUDAPlace(0)
    single_card_place = dev_list[0]
    start_exe = F.Executor(single_card_place)
    start_exe.run(startup_prog)

    #2. restore param
    if warm_start_setting is not None:
        if not os.path.exists(warm_start_setting.from_dir):
            raise ValueError('warm start dir not exists: %s' %
                             warm_start_setting.from_dir)
        log.info("warm start from %s" % warm_start_setting.from_dir)
        if warm_start_setting.predicate_fn is not None:

            def fn(v):
                ret = warm_start_setting.predicate_fn(v)
                if ret:
                    log.info('warm start: %s' % v.name)
                return ret

            F.io.load_vars(
                start_exe,
                warm_start_setting.from_dir,
                main_program=train_program,
                predicate=fn)
        else:
            raise NotImplementedError()

    saver = Saver(
        run_config.model_dir,
        start_exe,
        program=train_program,
        max_ckpt_to_keep=run_config.max_ckpt)
    if saver.last_ckpt is not None:
        train_init_state = saver.restore()
    else:
        train_init_state = None

    #3.create paralle executor(broadcast variable)
    train_exe = get_parallel_exe(train_program, model_spec.loss, len(dev_list))

    log.info('Device count %d' % F.core.get_cuda_device_count())
    #log.info('Memory usage per exapmle: %f' % F.contrib.memory_usage(program=train_program, batch_size=run_config.batch_size))

    try:  #[try -> with -> while]
        summary_writer = None
        if eval_dataset is not None:
            eval_summary_writers = {
                name:
                None  # summary wirter maybe none if tensorboard is not installed
                for name, ds in six.iteritems(eval_dataset)
            }
        else:
            eval_summary_writers = None
        try:
            from tensorboardX import SummaryWriter
            if distribution.status.is_master:
                summary_writer = SummaryWriter(
                    os.path.join(run_config.model_dir, 'train_history'))
                if eval_dataset is not None:
                    eval_summary_writers = {
                        name: SummaryWriter(
                            os.path.join(run_config.model_dir,
                                         os.path.join('eval_history', name)))
                        for name, ds in six.iteritems(eval_dataset)
                    }
        except ImportError:
            log.warning(
                'tensorboardX not installed, will not log to tensorboard')
        summary_record = SummaryRecord(
            scalar=collections.get(collection.Key.SUMMARY_SCALAR),
            histogram=collections.get(collection.Key.SUMMARY_HISTOGRAM), )

        train_run_hooks = [
            hooks.StopAtStepHook(
                run_config.max_steps, run_config.run_steps, msg='training'),
            hooks.LoggingHook(
                model_spec.loss,
                summary_record=summary_record,
                summary_writer=summary_writer,
                per_step=run_config.log_steps,
                skip_step=run_config.skip_steps),
        ]
        if distribution.status.is_master:
            train_run_hooks += [
                hooks.CheckpointSaverHook(
                    saver,
                    per_step=run_config.save_steps,
                    skip_step=run_config.skip_steps),
            ]

        train_run_hooks.extend(train_hooks)
        #initialize here to avoid creating one event file per run
        with MonitoredExecutor(
                train_exe,
                train_program,
                state=train_init_state,
                run_config=run_config,
                run_hooks=train_run_hooks, ) as train_exe:
            for data in train_dataset.start():
                train_exe.run(feed=data)  # train
                # start eval_loop
                if eval_dataset is not None and \
                    distribution.status.is_master and \
                    train_exe.state.gstep % run_config.eval_steps == 0 and \
                    train_exe.state.gstep > run_config.skip_steps:
                    eval_result = {}
                    for name, _ in six.iteritems(eval_dataset):
                        ret = evaluate(name, train_exe.state,
                                       eval_dataset[name],
                                       eval_summary_writers[name])
                        eval_result[name] = ret
                    for exporter in exporters:
                        exporter.export(start_exe, eval_program,
                                        eval_model_spec, eval_result,
                                        train_exe.state)
                    log.debug('eval done')
    except (F.core.EOFException, StopException):
        pass
    finally:
        if summary_writer is not None:
            summary_writer.close()
            if eval_summary_writers is not None:
                for v in eval_summary_writers.values():
                    if v is not None:
                        v.close()
