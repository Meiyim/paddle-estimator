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
import json
from functools import reduce
from time import time
import shutil

import logging
import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L

from atarashi import util
from atarashi.types import StopException
from . import distribution

log = logging.getLogger(__name__)

__all__ = ['MonitoredExecutor', 'Saver']


class RunState(object):
    @classmethod
    def from_str(cls, s):
        j = json.loads(s)
        ret = RunState()
        ret._gstep = j['global_step']
        ret._time = j['time']
        ret._step = 0
        return ret

    def __init__(self):
        self._gstep = 0
        self._step = 0
        self._time = time()

    @property
    def gstep(self):
        return self._gstep

    @property
    def step(self):
        return self._step

    @property
    def time(self):
        return self._time

    def __repr__(self):
        return repr({'global_step': self._gstep, 'time': self._time})

    def serialize(self):
        return json.dumps({'global_step': self._gstep, 'time': self._time})

    def next(self):
        ret = RunState()
        ret._gstep = self._gstep + 1
        ret._step = self._step + 1
        ret._time = time()
        return ret


class Saver(object):
    def __init__(self,
                 save_dir,
                 exe,
                 program,
                 save_prefix='model',
                 max_ckpt_to_keep=None):
        if exe is not None:
            assert isinstance(
                exe, F.Executor
            ), 'expect normal executor to save, got executor of type %s' % repr(
                type(exe))
        self._exe = exe
        self._program = program
        self._save_dir = save_dir
        self._save_prefix = save_prefix
        self._max_ckpt_to_keep = 10 if max_ckpt_to_keep is None else max_ckpt_to_keep

        self.ckpt_info_path = os.path.join(save_dir, 'ckpt_info')

        if os.path.exists(self.ckpt_info_path):
            self.ckpt_list = [
                p.strip() for p in open(self.ckpt_info_path).readlines()
            ]
            log.debug('ckpt_list in this Saver: %s' % (self.ckpt_list))
        else:
            self.ckpt_list = []

    @property
    def last_ckpt(self):
        return self.ckpt_list[-1] if len(self.ckpt_list) else None

    def save(self, state):
        save_name = '%s_%d' % (self._save_prefix, state.gstep)
        save_dir = os.path.join(self._save_dir, save_name)
        tmp_dir = os.path.join(self._save_dir, 'tmp')
        try:
            shutil.rmtree(save_dir)
            shutil.rmtree(tmp_dir)
        except FileNotFoundError:
            pass
        log.info('saving step %d to %s' % (state.gstep, save_dir))
        F.io.save_persistables(self._exe, tmp_dir, self._program)
        shutil.move(tmp_dir, save_dir)
        meta = state.serialize()
        open(os.path.join(save_dir, 'meta'), 'w').write(meta)

        self.ckpt_list.append(save_name)
        if len(self.ckpt_list) > self._max_ckpt_to_keep:
            ckpt_to_keep = self.ckpt_list[-self._max_ckpt_to_keep:]
            ckpt_to_remove = set(self.ckpt_list) - set(ckpt_to_keep)
            self.ckpt_list = ckpt_to_keep
            for ckpt in ckpt_to_remove:
                ckpt_dir = os.path.join(self._save_dir, ckpt)
                if os.path.exists(ckpt_dir):
                    shutil.rmtree(ckpt_dir)
                    log.debug('No. of ckpt exceed %d, clean up: %s' %
                              (self._max_ckpt_to_keep, ckpt_dir))
        open(self.ckpt_info_path, 'w').write('\n'.join(self.ckpt_list))

    def restore(self, ckpt=-1):
        assert abs(ckpt) <= len(
            self.ckpt_list), 'invalid restore ckpt number %d' % ckpt
        path = os.path.join(self._save_dir, self.ckpt_list[ckpt])
        meta_file = os.path.join(path, 'meta')
        if not os.path.exists(meta_file):
            raise RuntimeError('meta not found in restore dir: %s' % path)
        state = RunState.from_str(open(meta_file).read())
        log.info('restore from ckpt %s, ckpt-status: %s' % (path, repr(state)))

        def fn(v):
            vpath = os.path.join(path, v.name)
            if F.io.is_persistable(v):
                if os.path.exists(vpath):
                    return True
                else:
                    log.warning('var %s not found in checkpoint, ignored' %
                                v.name)
            return False

        F.io.load_vars(
            self._exe, path, main_program=self._program, predicate=fn)
        return state


class MonitoredExecutor(object):
    """A wrapper handling the train loop"""

    def __init__(self,
                 executor,
                 program,
                 state=None,
                 run_config=None,
                 run_hooks=[]):
        self._exe = executor
        self._hooks = run_hooks
        if isinstance(executor, F.ParallelExecutor):
            self._dev_count = executor.device_count
        else:
            self._dev_count = 1
        self._state = RunState() if state is None else state

        self._program = program
        if run_config is not None:
            self._save_dir = run_config.model_dir
            self._save_steps = run_config.save_steps
            self._skip_step = run_config.skip_steps if run_config.skip_steps else 100
            self._save_prefix = 'model'

    @property
    def state(self):
        return self._state

    def __enter__(self):
        log.info('********** Start Loop ************')
        # TODO init

        for h in self._hooks:
            log.debug('train loop has hook %s' % h)
            h.before_train()
        return self

    def run(self, fetch_list=[], *args, **kwargs):
        #log.debug('Executor running step %d' % self._state.gstep)
        if self._hooks:
            fetch_list = [h.before_run(self._state)
                          for h in self._hooks] + fetch_list
            fetch_list_len = map(len, fetch_list)

            fetch_list, schema = util.flatten(fetch_list)
            #if len(set(fetch_list)) != len(fetch_list):
            #    log.error('strange shit happend when fetch list has idetity tensors %s' % fetch_list)

            #log.debug(fetch_list)
            if isinstance(self._exe, F.ParallelExecutor):
                res = self._exe.run(fetch_list=fetch_list, *args, **kwargs)
            else:
                res = self._exe.run(self._program,
                                    fetch_list=fetch_list,
                                    *args,
                                    **kwargs)
            if self._dev_count > 1:
                res = [self.merge_result(r) for r in res]
            #log.debug(res)

            res = util.unflatten(res, schema)
            for r, h in zip(res, self._hooks):
                h.after_run(r, self._state)

            if any(map(lambda i: i.should_stop(self._state), self._hooks)):
                raise StopException('hook call stop')

        #if self._start_exe is not None and \
        #    self._state.gstep % self._save_steps == 0 and \
        #    self._state.step > self._skip_step:
        #    self._save_ckpt()
        self._state = self._state.next()

    def __exit__(self, err_type, err_value, trace):
        if (err_type is None) or (err_type is F.core.EOFException) or (
                err_type is StopException) or (err_type is KeyboardInterrupt):
            try:
                log.info('********** Stop Loop ************')
                for h in self._hooks:
                    h.after_train()
            except Exception as e:
                log.exception('error occur after loop %s' % repr(e))
                raise e
        else:
            log.info('********** Interupt Loop ************')
            log.exception('error occur during loop %s: %s' %
                          (err_type, err_value))

    def merge_result(self, ls):
        shape = (-1, ls.shape[0] // self._dev_count) + ls.shape[1:]
        ret = np.reshape(ls, shape).mean(axis=0)
        return ret
