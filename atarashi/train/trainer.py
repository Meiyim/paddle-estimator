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

import os
import itertools
import inspect
from contextlib import contextmanager

import paddle.fluid  as F
import paddle.fluid.layers  as L

import atarashi
import atarashi.collection
from atarashi.types import RunMode, StopException, SummaryRecord
import atarashi.train
from atarashi import metrics
from atarashi import summary
from atarashi.train import hooks, MonitoredExecutor

from atarashi import log

__all__ = ['train_and_eval'] 


def get_parallel_exe(program, loss, dev_count):
    nccl2_num_trainers = 1
    nccl2_trainer_id = 0

    exec_strategy = F.ExecutionStrategy()
    exec_strategy.num_threads = dev_count
    exec_strategy.num_iteration_per_drop_scope = min(10, 1000) # important shit

    build_strategy = F.BuildStrategy()
    build_strategy.remove_unnecessary_lock = False

    train_exe = F.ParallelExecutor(
        use_cuda=True,
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy,
        main_program=program,
        num_trainers=nccl2_num_trainers,
        trainer_id=nccl2_trainer_id)
    return train_exe


def build_net(model_fn_or_model, dataset, mode, params, run_config):
    if isinstance(model_fn_or_model, atarashi.train.Model):
        def model_fn(features, mode, params, run_config):
            fea, label = features[: -1], features[-1]
            model = model_fn_or_model(params, mode, params, run_config=run_config)
            pred = model.forward(fea)
            loss = model.loss(pred, label)
            model.backward(loss)
    elif inspect.isfunction(model_fn_or_model):
        model_fn = model_fn_or_model
    else:
        raise ValueError('unknown model %s' % model_fn_or_model)

    fea = dataset.features()
    model_spec = model_fn(features=fea, mode=mode, params=params, run_config=run_config)
    if mode is RunMode.TRAIN or mode is RunMode.EVAL:
        assert model_spec.loss is not None
    elif mode is RunMode.PREDICT:
        assert model_spec.predictions is not None
    else:
        raise ValueError('unkonw mode %s' % mode)
    return model_spec


def train_and_eval(model_class_or_model_fn, params, run_config, train_dataset, eval_dataset=None, warm_start_setting=None, train_hooks=[], eval_hooks=[], exporters=[]):
    train_program = F.Program()
    startup_prog = F.Program()
    with F.program_guard(train_program, startup_prog):
        with F.unique_name.guard():
            with atarashi.collection.Collections() as collections:
                model_spec = build_net(model_class_or_model_fn, train_dataset, RunMode.TRAIN, params, run_config)

    if eval_dataset is not None:
        eval_program = F.Program()
        eval_startup_program = startup_prog
        with F.program_guard(eval_program, eval_startup_program):
            #share var with Train net
            with F.unique_name.guard():
                eval_model_spec = build_net(model_class_or_model_fn, eval_dataset, RunMode.EVAL, params, run_config)

    dev_count = F.core.get_cuda_device_count()
    #param broadcast happened when creating ParallelProgram, init before this

    #The order of this 3 steps really matters
    #1. init train
    start_exe = F.Executor(F.CUDAPlace(0))
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
            F.io.load_vars(start_exe, warm_start_setting.from_dir, main_program=train_program, predicate=fn)
        else:
            raise NotImplementedError()

    saver = atarashi.train.Saver(run_config.model_dir, start_exe, program=train_program)
    if saver.last_ckpt is not None:
        train_init_state = saver.restore()
    else:
        train_init_state = None

    #3.create paralle executor(broadcast variable)
    train_exe = get_parallel_exe(train_program, model_spec.loss, dev_count)

    log.info('Device count %d' % F.core.get_cuda_device_count())
    #log.info('Memory usage per exapmle: %f' % F.contrib.memory_usage(program=train_program, batch_size=run_config.batch_size))


    try: #[try -> with -> while]
        summary_record = SummaryRecord(
                scalar=collections.get_from(summary.KEY_SUMMARY_SCALAR),
                histogram=collections.get_from(summary.KEY_SUMMARY_HISTOGRAM),
            )

        train_run_hooks = [
                hooks.CheckpointSaverHook(saver, per_step=run_config.save_steps, skip_step=run_config.skip_steps),
                hooks.LoggingHook(model_spec.loss, board_log_dir=os.path.join(run_config.model_dir, 'train_history'), 
                    summary_record=summary_record,
                    per_step=run_config.log_steps, 
                    skip_step=run_config.skip_steps),
                hooks.StopAtStepHook(run_config.max_steps, run_config.run_steps),
            ]
        train_run_hooks.extend(train_hooks)
        #initialize here to avoid creating one event file per run
        eval_hook = hooks.EvalHook(eval_model_spec.metrics, board_log_dir=os.path.join(run_config.model_dir, 'eval_history')) 

        with train_dataset.start(), \
            MonitoredExecutor(train_exe, 
               train_program,
               state=train_init_state,
               run_config=run_config,
               dev_count=dev_count,
               run_hooks=train_run_hooks,
            ) as train_exe:
            while True:
                train_exe.run() # train
                #start eval_loop
                if eval_dataset and \
                    train_exe.state.gstep % run_config.eval_steps == 0 and \
                    train_exe.state.gstep > run_config.skip_steps:
                    try: #[try -> with -> while]
                        eval_hook.set_train_state(train_exe.state)
                        eval_run_hooks = [eval_hook]
                        eval_run_hooks.extend(eval_hooks)
                        with eval_dataset.start(), \
                            MonitoredExecutor(start_exe,
                               program=eval_program,
                               run_config=None,
                               dev_count=1, # single card eval
                               run_hooks=eval_run_hooks,
                            ) as eval_exe:
                            while True:
                                eval_exe.run()
                                #log.debug('eval')
                    except (F.core.EOFException, StopException):
                        pass
                    eval_result = eval_hook.result
                    for exporter in exporters:
                        pass
                        #exporter.export(start_exe, train_program, eval_result, train_exe.state, [f.name for f in features], [eval_pred])
                    log.debug('eval done')
    except (F.core.EOFException, StopException):
        pass


