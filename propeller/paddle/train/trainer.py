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
from collections import namedtuple
from contextlib import contextmanager
from six.moves import zip, map
import logging
from time import time

import paddle.fluid as F
import paddle.fluid.layers as L

from propeller.types import RunMode, StopException, SummaryRecord, StopException, ModelSpec, InferenceSpec, ProgramPair, RunConfig
from propeller.paddle import summary, collection
from propeller.paddle.data.functional import Dataset
from propeller.paddle.train import distribution
from propeller.train.model import Model
from propeller.paddle.train.monitored_executor import Saver
from propeller.paddle.train import hooks, metrics

from propeller.paddle.train.monitored_executor import MonitoredExecutor

log = logging.getLogger(__name__)

__all__ = ['train_and_eval', 'Estimator']


def get_parallel_exe(program, loss, dev_count):
    exec_strategy = F.ExecutionStrategy()
    exec_strategy.num_threads = 4  #2 for fp32 4 for fp16
    exec_strategy.use_experimental_executor = True
    exec_strategy.num_iteration_per_drop_scope = 10  #important shit

    build_strategy = F.BuildStrategy()
    build_strategy.remove_unnecessary_lock = False
    build_strategy.memory_optimize = True
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


def build_net(model_fn, features, mode, params, run_config):
    model_spec = model_fn(
        features=features, mode=mode, params=params, run_config=run_config)
    if not isinstance(model_spec.predictions, list):
        raise ValueError('model_spec.predictions shuold be list, got %s' %
                         repr(model_spec.predictions))

    if mode == RunMode.TRAIN:
        if not isinstance(model_spec.loss, F.framework.Variable):
            raise ValueError('model_spec.metrics should be Variable, got %s' %
                             repr(model_spec.loss))
    elif mode == RunMode.EVAL:
        if not isinstance(model_spec.metrics, dict):
            raise ValueError('model_spec.metrics should be dict, got %s' %
                             repr(model_spec.metrics))
    elif mode == RunMode.PREDICT:
        assert model_spec.predictions is not None
    else:
        raise ValueError('unkonw mode %s' % mode)
    return model_spec


class Estimator(object):
    def __init__(self,
                 model_class_or_model_fn,
                 run_config,
                 params=None,
                 warm_start_setting=None):
        '''
        model_class_or_model_fn(callable|propeller.train.Model): `model_class_or_model_fn` be specified in 2 ways:
            1. subclass of propeller.train.Model which implements:
                1. \_\_init\_\_       (hyper_param, mode, run_config)
                2. forward            (features) => (prediction)
                3. backword           (loss) => None
                4. loss               (predictoin) => (loss)
                5. metrics (optional) (prediction) => (dict of propeller.Metrics)
                
            2. a model_fn takes following args:
                1. features
                2. param
                3. mode
                4. run_config(optional)
               and returns a `propeller.ModelSpec`

        params: any python object, will pass to your `model_fn` or `propeller.train.Model`
        run_config (propeller.RunConfig): run_config.max_steps should not be None.
        warm_start_setting (propeller.WarmStartSetting): Optional. warm start variable will overwrite model variable.
        '''
        if run_config.model_dir is None:
            raise ValueError('model_dir should specified in run_config')

        if issubclass(model_class_or_model_fn, Model):

            def model_fn(features, mode, params, run_config):
                if mode != RunMode.PREDICT:
                    fea, label = features[:-1], features[-1]
                else:
                    fea = features

                model = model_class_or_model_fn(
                    params, mode, run_config=run_config)
                pred = model.forward(fea)
                if isinstance(pred, F.framework.Variable):
                    prediction = [pred]
                else:
                    prediction = pred
                if mode == RunMode.TRAIN:
                    loss = model.loss(pred, label)
                    model.backward(loss)
                    return ModelSpec(
                        loss=loss, predictions=prediction, mode=mode)
                elif mode == RunMode.EVAL:
                    loss = model.loss(pred, label)
                    me = model.metrics(pred, label)

                    inf_spec = InferenceSpec(inputs=fea, outputs=prediction)
                    if 'loss' not in me:
                        me['loss'] = metrics.Mean(loss)
                    return ModelSpec(
                        loss=loss,
                        predictions=prediction,
                        metrics=me,
                        mode=mode,
                        inference_spec=inf_spec)
                elif mode == RunMode.PREDICT:
                    return ModelSpec(predictions=prediction, mode=mode)
                else:
                    raise RuntimeError('unknown run mode %s' % mode)
        elif inspect.isfunction(model_class_or_model_fn):
            model_fn = model_class_or_model_fn
        else:
            raise ValueError('unknown model %s' % model_class_or_model_fn)

        self.model_fn = model_fn
        self.params = params
        self.run_config = run_config
        self.warm_start_setting = warm_start_setting

    def build_for_train(self, train_dataset):
        train_dataset.name = 'train'
        train_program = F.Program()
        startup_prog = F.Program()
        with F.program_guard(train_program, startup_prog):
            with F.unique_name.guard():
                with collection.Collections() as collections:
                    log.info('Building Train Graph...')
                    fea = train_dataset.features()
                    model_spec = build_net(self.model_fn, fea, RunMode.TRAIN,
                                           self.params, self.run_config)
                    log.info('Building Train Graph: Done')

                scalars = collections.get(collection.Key.SUMMARY_SCALAR)
                histograms = collections.get(collection.Key.SUMMARY_HISTOGRAM)
                skip_optimize_ops = collections.get(
                    collection.Key.SKIP_OPTIMIZE)
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
            % (repr(self.run_config), repr(self.params), repr(model_spec)))

        summary_record = SummaryRecord(
            scalar=collections.get(collection.Key.SUMMARY_SCALAR),
            histogram=collections.get(collection.Key.SUMMARY_HISTOGRAM), )
        return ProgramPair(
            train_program=train_program,
            startup_program=startup_prog), model_spec, summary_record

    def build_for_eval(self, ds):
        ds.name = 'eval'
        program = F.Program()
        startup_prog = F.Program()
        with F.program_guard(program, startup_prog):
            #share var with Train net
            with F.unique_name.guard():
                log.info('Building Eval Graph')
                fea = ds.features()
                model_spec = build_net(self.model_fn, fea, RunMode.EVAL,
                                       self.params, self.run_config)
                log.info('Done')
        program = program.clone(for_test=True)
        log.info(
            'Eval with: \n> Run_config: %s\n> Params: %s\n> Train_model_spec: %s\n'
            % (repr(self.run_config), repr(self.params), repr(model_spec)))
        return ProgramPair(
            train_program=program, startup_program=startup_prog), model_spec

    def build_for_predict(self, ds):
        ds.name = 'predict'
        program = F.Program()
        startup_prog = F.Program()
        with F.program_guard(program, startup_prog):
            #share var with Train net
            with F.unique_name.guard():
                log.info('Building Predict Graph')
                fea = ds.features()
                model_spec = build_net(self.model_fn, fea, RunMode.PREDICT,
                                       self.params, self.run_config)
                log.info('Done')

        program = program.clone(for_test=True)

        log.info(
            'Predict with: \n> Run_config: %s\n> Params: %s\n> Train_model_spec: %s\n'
            % (repr(self.run_config), repr(self.params), repr(model_spec)))
        return ProgramPair(
            train_program=program, startup_program=startup_prog), model_spec

    def train(self, train_ds, train_hooks=[]):
        if not isinstance(train_ds, Dataset):
            raise ValueError('expect dataset to be instance of Dataset, got %s'
                             % repr(train_ds))

        summary_writer = None
        try:
            from tensorboardX import SummaryWriter
            if distribution.status.is_master:
                summary_writer = SummaryWriter(
                    os.path.join(self.run_config.model_dir, 'train_history'))
        except ImportError:
            log.warning(
                'tensorboardX not installed, will not log to tensorboard')

        train_program, model_spec, summary_record = self.build_for_train(
            train_ds)
        train_run_hooks = [
            hooks.StopAtStepHook(self.run_config.max_steps,
                                 self.run_config.run_steps),
            hooks.LoggingHook(
                model_spec.loss,
                summary_record=summary_record,
                summary_writer=summary_writer,
                per_step=self.run_config.log_steps,
                skip_step=self.run_config.skip_steps),
        ]
        train_run_hooks.extend(train_hooks)
        train_executor = F.Executor(F.cuda_places()[0])

        mon_exe = MonitoredExecutor(
            train_executor,
            train_program,
            run_config=self.run_config,
            run_hooks=train_run_hooks, )

        mon_exe.init_or_restore_variables()
        if distribution.status.is_master:
            mon_exe._hooks.append(
                hooks.CheckpointSaverHook(
                    mon_exe._saver,
                    per_step=mon_exe._save_steps,
                    skip_step=mon_exe._skip_steps))

        with mon_exe:
            for data in train_ds.start():
                mon_exe.run(feed=data)

    def evaluate(self, eval_dataset, eval_hooks=[]):
        if not isinstance(eval_dataset, Dataset):
            raise ValueError('expect dataset to be instance of Dataset, got %s'
                             % repr(eval_dataset))
        program, model_spec = self.build_for_eval(eval_dataset)
        single_card_place = F.cuda_places()[0]
        eval_executor = F.Executor(single_card_place)
        eval_hooks = [
            hooks.EvalHook(
                'eval',
                model_spec.metrics,
                summary_writer=None, )
        ]

        mon_exe = MonitoredExecutor(
            eval_executor,
            program,
            run_config=self.run_config,
            run_hooks=eval_hooks)
        mon_exe.init_or_restore_variables()

        with mon_exe:
            for data in eval_dataset.start(places=[single_card_place]):
                mon_exe.run(feed=data)

    def predict(self, predict_dataset, ckpt=None, steps=-1, split_batch=True):
        '''
        Perform predictoin
        will call `model_fn` and initiate user-specifed model in `propeller.RunMode.PREDICT` mode 

        Args:
            infer_dataset (propeller.data.Dataset): should not `shuffle` or `repeat`
            steps (int): steps to predict, if -1 is specifed, will stop when `StopException` is raised in `infer_dataset`
            split_batch (bool): if True, prediction of each example in a batch is returned.

        Yields:
            Evaluated values of predictions tensors.

        '''
        if not isinstance(predict_dataset, Dataset):
            raise ValueError('expect dataset to be instance of Dataset, got %s'
                             % repr(train_ds))

        program, model_spec = self.build_for_predict(predict_dataset)
        single_card_place = F.cuda_places()[0]
        executor = F.Executor(single_card_place)
        pred_run_config = RunConfig(
            run_steps=steps if steps == -1 else None,
            model_dir=self.run_config.model_dir)
        mon_exe = MonitoredExecutor(
            executor,
            program,
            run_config=pred_run_config, )
        mon_exe.init_or_restore_variables()
        with mon_exe:
            mon_exe._state = mon_exe._saver.restore(ckpt)
            for data in predict_dataset.start(places=[single_card_place]):
                mon_exe.run(feed=data)
            log.info('Runining predict from dir: %s' % repr(mon_exe.state))
            single_card_place = F.cuda_places()[0]
            for data in predict_dataset.start(places=[single_card_place]):
                res = mon_exe.run(fetch_list=model_spec.predictions, feed=data)
                if split_batch:
                    res = map(lambda i: i.tolist(), res)
                    res = zip(*res)  # transpose
                    for r in res:
                        yield r
                else:
                    yield res


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
    will call `model_fn` and initiate user-specifed model in `propeller.RunMode.PREDICT` mode 

    Args:
        model_class_or_model_fn(callable|propeller.train.Model): `model_class_or_model_fn` be specified in 2 ways:
            1. subclass of propeller.train.Model which implements:
                1. \_\_init\_\_       (hyper_param, mode, run_config)
                2. forward            (features) => (prediction)
                3. backword           (loss) => None
                4. loss               (predictoin) => (loss)
                5. metrics (optional) (prediction) => (dict of propeller.Metrics)
                
            2. a model_fn takes following args:
                1. features
                2. param
                3. mode
                4. run_config(optional)
               and returns a `propeller.ModelSpec`

        params: any python object, will pass to your `model_fn` or `propeller.train.Model`
        run_config (propeller.RunConfig): run_config.max_steps should not be None.
        train_dataset (propeller.paddle.data.Dataset): training will stop if global_step > run_config.max_steps.
        eval_dataset (propeller.paddle.data.Dataset|dict): Optional, if Dict of propeller.data.Dataset were specified, will perform evluatation on every evaluation sets and report results.
        warm_start_setting (propeller.WarmStartSetting): Optional. warm start variable will overwrite model variable.
        train_hooks (list of propeller.paddle.train.RunHook): Optional.
        eval_hooks (list of propeller.paddle.train.RunHook): Optional.
        exporters (list of propeller.paddle.train.Exporter): Optional.
    '''
    if _shit is not None:
        raise ValueError('specify keyword args to this function')
    if model_class_or_model_fn is None or params is None or run_config is None or train_dataset is None:
        raise ValueError(
            'some argument is None: model_class_or_model_fn:%s params:%s run_config:%s train_dataset:%s'
            % (model_class_or_model_fn, params, run_config, train_dataset))

    #init distribution env if envvir PROPELLER_DISCONFIG is set
    distribution.init_distribuition_env(train_program, startup_prog)

    if eval_dataset is not None:
        if not isinstance(eval_dataset, (dict, Dataset)):
            raise ValueError(
                'Eval dataset should be propeller.Dataset of a list of that, got: %s'
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
