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
from contextlib import contextmanager

import paddle.fluid  as F
import paddle.fluid.layers  as L

import atarashi
import atarashi.collection
from atarashi.types import RunMode, StopException, SummaryRecord
from atarashi.train import hooks, MonitoredExecutor, bind_pyreader
from atarashi import metrics
from atarashi import summary

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


def train_and_eval(Model, params, run_config, train_dataset, eval_dataset=None, warm_start_setting=None, train_hooks=[], eval_hooks=[], exporters=[]):
    train_program = F.Program()
    startup_prog = F.Program()
    with F.program_guard(train_program, startup_prog):
        with F.unique_name.guard():
            with atarashi.collection.Collections() as collections:
                pyreader = L.py_reader(
                        50,
                        shapes=train_dataset.output_shapes,
                        dtypes=train_dataset.output_types,
                        lod_levels=[0] * len(eval_dataset.output_shapes),
                        name='train',
                        use_double_buffer=True,
                )
                features = L.read_file(pyreader)
                label = features[-1]
                features = features[: -1]
                model = Model(params, RunMode.TRAIN)
                pred = model.forward(features)
                loss = model.loss(pred, label)
                model.backward(loss)

    if eval_dataset is not None:
        eval_program = F.Program()
        eval_startup_program = startup_prog
        with F.program_guard(eval_program, eval_startup_program):
            #share var with Train net
            with F.unique_name.guard():
                eval_pyreader = L.py_reader(
                    50,
                    shapes=eval_dataset.output_shapes,
                    dtypes=eval_dataset.output_types,
                    lod_levels=[0] * len(eval_dataset.output_shapes),
                    name='eval',
                    use_double_buffer=True,
                )
                features = L.read_file(eval_pyreader)
                label = features[-1]
                features = features[: -1]

                eval_model = Model(params, RunMode.EVAL)
                eval_pred = eval_model.forward(features)
                #eval_loss = eval_model.loss(eval_pred, label)
                eval_metrics = eval_model.metrics(eval_pred, label)
                #assert 'loss' not in eval_metrics
                #eval_metrics['loss'] = metrics.Mean(eval_loss)

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
    train_exe = get_parallel_exe(train_program, loss, dev_count)

    log.info('Device count %d' % F.core.get_cuda_device_count())
    #log.info('Memory usage per exapmle: %f' % F.contrib.memory_usage(program=train_program, batch_size=run_config.batch_size))


    try: #[try -> with -> while]
        summary_record = SummaryRecord(
                scalar=collections.get_from(summary.KEY_SUMMARY_SCALAR),
                histogram=collections.get_from(summary.KEY_SUMMARY_HISTOGRAM),
            )

        train_run_hooks = [
                hooks.CheckpointSaverHook(saver, per_step=run_config.save_steps, skip_step=run_config.skip_steps),
                hooks.LoggingHook(loss, board_log_dir=os.path.join(run_config.model_dir, 'train_history'), 
                    summary_record=summary_record,
                    per_step=run_config.log_steps, 
                    skip_step=run_config.skip_steps),
                hooks.StopAtStepHook(run_config.max_steps, run_config.run_steps),
            ]
        train_run_hooks.extend(train_hooks)
        #initialize here to avoid creating one event file per run
        eval_hook = hooks.EvalHook(eval_metrics, board_log_dir=os.path.join(run_config.model_dir, 'eval_history')) 

        with bind_pyreader(pyreader, train_dataset), \
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
                        with bind_pyreader(eval_pyreader, eval_dataset), \
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


