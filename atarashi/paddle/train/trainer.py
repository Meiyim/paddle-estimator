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

import paddle.fluid as F
import paddle.fluid.layers as L

from atarashi.types import RunMode, StopException, SummaryRecord, StopException, ModelSpec
from atarashi.paddle import summary, collection
from atarashi.paddle.data.functional import Dataset
from atarashi.paddle.train import distribution
from atarashi.train.model import Model
from atarashi.paddle.train.monitored_executor import Saver
from atarashi.paddle.train import hooks, metrics

from atarashi.paddle.train.monitored_executor import MonitoredExecutor

from atarashi import log

__all__ = ['train_and_eval', 'predict']


def get_parallel_exe(program, loss, dev_count):
    exec_strategy = F.ExecutionStrategy()
    exec_strategy.num_threads = 4  #2 for fp32 4 for fp16
    exec_strategy.use_experimental_executor = True
    exec_strategy.num_iteration_per_drop_scope = 10  #important shit

    build_strategy = F.BuildStrategy()
    build_strategy.remove_unnecessary_lock = False

    log.debug('replica id %d of %d' % (distribution.status.replica_id,
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
                if 'loss' not in me:
                    me['loss'] = metrics.Mean(loss)
                return ModelSpec(
                    loss=loss, predictions=pred, metrics=me, mode=mode)
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
        assert model_spec.metrics is not None and model_spec.loss is not None
    elif mode == RunMode.PREDICT:
        assert model_spec.predictions is not None
    else:
        raise ValueError('unkonw mode %s' % mode)
    return model_spec


def predict(model_class_or_model_fn,
            params,
            model_dir,
            infer_dataset,
            run_config=None,
            steps=-1,
            split_batch=True):
    if not os.path.exists(model_dir):
        raise ValueError('model dir not found %s' % model_dir)

    program = F.Program()
    startup_prog = F.Program()
    with F.program_guard(program, startup_prog):
        with F.unique_name.guard():
            fea = infer_dataset.features()
            model_spec = build_net(model_class_or_model_fn, fea,
                                   RunMode.PREDICT, params, run_config)
    program = program.clone(for_test=True)
    start_exe = F.Executor(F.CUDAPlace(0))
    start_exe.run(startup_prog)

    F.io.load_vars(
        start_exe,
        model_dir,
        main_program=program,
        predicate=F.io.is_persistable)

    pred = model_spec.predictions
    if not isinstance(pred, list) and not isinstance(pred, tuple):
        pred = [pred]
    try:
        log.info('Runining predict from dir: %s' % model_dir)
        with infer_dataset.start():
            for _ in itertools.count(steps):
                res = start_exe.run(program, fetch_list=pred)
                if split_batch:
                    res = map(lambda i: i.tolist(), res)
                    res = zip(*res)  # transpose
                    for r in res:
                        yield r
                else:
                    yield res
    except F.core.EOFException:
        log.debug('Predict done')


def train_and_eval(model_class_or_model_fn,
                   params,
                   run_config,
                   train_dataset,
                   eval_dataset=None,
                   warm_start_setting=None,
                   train_hooks=[],
                   eval_hooks=[],
                   exporters=[]):
    train_program = F.Program()
    startup_prog = F.Program()
    with F.program_guard(train_program, startup_prog):
        with F.unique_name.guard():
            with collection.Collections() as collections:
                log.info('Building Train Graph')
                fea = train_dataset.features()
                model_spec = build_net(model_class_or_model_fn, fea,
                                       RunMode.TRAIN, params, run_config)

            scalars = collections.get_from(summary.KEY_SUMMARY_SCALAR)
            histograms = collections.get_from(summary.KEY_SUMMARY_HISTOGRAM)
            skip_opt = set()
            if scalars is not None:
                skip_opt |= {t for _, t in scalars}
            if histograms is not None:
                skip_opt |= {t for _, t in histograms}
            F.memory_optimize(
                input_program=train_program, skip_opt_set=list(skip_opt))
            log.info('Done')

    log.debug(
        'Train with: \n> Run_config: %s\n> Params: %s\n> Train_model_spec: %s\n'
        % (repr(run_config), repr(params), repr(model_spec)))

    #init distribution env if envvir ATARASHI_DISCONFIG is set
    distribution.init_distribuition_env(train_program, startup_prog)

    if eval_dataset is not None:
        if not (isinstance(eval_dataset, dict) or isinstance(eval_dataset,
                                                             Dataset)):
            raise ValueError(
                'Eval dataset should be atarashi.Dataset of a list of that, got: %s'
                % eval_dataset)
        if isinstance(eval_dataset, Dataset):
            eval_dataset = {'eval': eval_dataset}
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

        def evaluate(name, train_state):
            try:  #[try -> with -> while]
                ds = eval_dataset[name]
                program = eval_program[name]
                log_dir = os.path.join(
                    os.path.join(run_config.model_dir, 'eval_history'), name)
                eval_hook = hooks.EvalHook(
                    name, eval_model_spec.metrics, board_log_dir=log_dir)
                eval_hook.set_train_state(train_state)
                eval_run_hooks = [eval_hook]
                eval_run_hooks.extend(eval_hooks)
                with ds.start(), \
                    MonitoredExecutor(start_exe,
                       program=program,
                       run_config=None,
                       run_hooks=eval_run_hooks,
                    ) as eval_exe:
                    while True:
                        eval_exe.run()
                        #log.debug('eval')
            except (F.core.EOFException, StopException):
                log.debug('Eval dataset ran out of data')
                return eval_hook.result

        log.debug('Eval with: \n> Eval_model_spec %s' % repr(eval_model_spec))

    dev_list = F.cuda_places()  #list all visible divices
    log.debug('Visible device %s' % repr(dev_list))
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
        log.info("warm start from %s" % warm_start_setting.from_dir)
        if warm_start_setting.predicate_fn is not None:

            def fn(v):
                ret = warm_start_setting.predicate_fn(v)
                if ret:
                    log.debug('warm start: %s' % v.name)
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
        summary_record = SummaryRecord(
            scalar=collections.get_from(summary.KEY_SUMMARY_SCALAR),
            histogram=collections.get_from(summary.KEY_SUMMARY_HISTOGRAM), )

        train_run_hooks = [
            hooks.StopAtStepHook(run_config.max_steps, run_config.run_steps),
            hooks.LoggingHook(
                model_spec.loss,
                board_log_dir=os.path.join(run_config.model_dir,
                                           'train_history'),
                summary_record=summary_record,
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
        with train_dataset.start(), \
            MonitoredExecutor(train_exe,
               train_program,
               state=train_init_state,
               run_config=run_config,
               run_hooks=train_run_hooks,
            ) as train_exe:
            while True:
                train_exe.run()  # train
                # start eval_loop
                if eval_dataset is not None and \
                    train_exe.state.gstep % run_config.eval_steps == 0 and \
                    train_exe.state.gstep > run_config.skip_steps:
                    eval_result = {}
                    for name, _ in six.iteritems(eval_dataset):
                        ret = evaluate(name, train_exe.state)
                        eval_result[name] = ret
                    for exporter in exporters:
                        exporter.export(start_exe, train_program, eval_result,
                                        train_exe.state)
                    log.debug('eval done')
    except (F.core.EOFException, StopException):
        log.debug('Train dataset ran out of data')
