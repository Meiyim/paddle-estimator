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

import sys
import six
import os
import itertools

import numpy as np
import logging
import paddle.fluid as F
import paddle.fluid.layers as L
from tensorboardX import SummaryWriter

from atarashi import util
from atarashi.paddle.train import distribution
from atarashi.paddle.train.metrics import Metrics

log = logging.getLogger(__name__)


class RunHook(object):
    def __init__(self):
        pass

    def before_train(self):
        pass

    def before_run(self, state):
        return []

    def after_run(self, res_list, state):
        pass

    def should_stop(self, state):
        return False

    def after_train(self):
        pass


class LoggingHook(RunHook):
    def __init__(self,
                 loss,
                 per_step=10,
                 skip_step=100,
                 board_log_dir=None,
                 summary_record=None):
        self.loss = loss.name
        self.per_step = per_step
        self.skip_step = skip_step
        self.board_log_dir = board_log_dir
        self.summary_record = summary_record
        self.writer = None
        self.last_state = None

    def before_train(self):
        if self.summary_record:
            assert self.board_log_dir
            if self.summary_record.scalar:
                self.s_name, self.s_tolog = zip(*self.summary_record.scalar)
            else:
                self.s_name, self.s_tolog = [], []

            if self.summary_record.histogram:
                self.h_name, self.h_tolog = zip(*self.summary_record.histogram)
            else:
                self.h_name, self.h_tolog = [], []

        if distribution.status.is_master:
            self.writer = SummaryWriter(self.board_log_dir)
        else:
            self.writer = None

    def before_run(self, state):
        if state.gstep % self.per_step == 0 and state.step > self.skip_step:
            ret = [self.loss]
            if self.summary_record:
                ret += self.s_tolog
                ret += self.h_tolog
            return ret
        else:
            return []

    def after_run(self, res_list, state):
        if state.gstep % self.per_step == 0 and state.step > self.skip_step:
            if not self.summary_record:
                return

            loss = float(res_list[0])
            s_np = res_list[1:1 + len(self.s_name)]
            h_np = res_list[1 + len(self.s_name):1 + len(self.s_name) + len(
                self.h_name)]

            if self.last_state is not None:
                speed = (state.gstep - self.last_state.gstep) / (
                    state.time - self.last_state.time)
            else:
                speed = -1.
            self.last_state = state

            # log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar('loss', loss, state.gstep)
                for name, t in zip(self.s_name, s_np):
                    if np.isnan(t).any():
                        log.warning('Nan summary: %s, skip' % name)
                    else:
                        self.writer.add_scalar(name, t, state.gstep)

                for name, t in zip(self.h_name, h_np):
                    if np.isnan(t).any():
                        log.warning('Nan summary: %s, skip' % name)
                    else:
                        self.writer.add_histogram(name, t, state.gstep)

                if speed > 0.:
                    self.writer.add_scalar('global_step', speed, state.gstep)

            # log to stdout
            log.debug('\t'.join([
                'step: %d' % state.gstep,
                'steps/sec: %.5f' % speed,
                'loss: %.5f' % loss,
                '' if self.summary_record is None else ' '.join(
                    map(lambda t: '%s:%s' % t, zip(self.s_name, s_np))),
            ]))

    def after_train(self):
        if self.writer:
            self.writer.close()


class StopAtStepHook(RunHook):
    def __init__(self, stop_global_step, stop_step, msg=None):
        self._stop_gstep = stop_global_step
        self._stop_step = stop_step

        try:
            import tqdm
            self._tqdm = tqdm.tqdm(total=stop_global_step, desc=msg)
        except ImportError:
            log.warning('tqdm not installed, will not show progress bar')
            self._tqdm = None

    def before_run(self, state):
        if self._tqdm is not None:
            self._tqdm.n = state.gstep
            self._tqdm.refresh()
        return []

    def after_train(self):
        if self._tqdm is not None:
            self._tqdm.close()

    def should_stop(self, state):
        if (self._stop_gstep and state.gstep >= self._stop_gstep) or \
           (self._stop_step and state.step >= self._stop_step):
            log.info('StopAtStepHook called stop')
            return True
        else:
            return False


class EvalHook(RunHook):
    """hook this on a eval Executor"""

    def __init__(self, name, metrics, board_log_dir):
        self.board_log_dir = board_log_dir
        self._name = name
        self.train_state = None
        self.writer = None
        self._result = None

        if not isinstance(metrics, dict):
            raise ValueError('metrics should be dict, got %s' % repr(metrics))

        for k, m in six.iteritems(metrics):
            if not isinstance(m, Metrics):
                raise ValueError(
                    'metrics %s should be instance of atarashi.Metrics, got %s'
                    % (k, repr(m)))

        if len(metrics):
            self.names = list(metrics.keys())
            self.metrics = list(metrics.values())
        else:
            self.names, self.metrics = [], []
        self.writer = SummaryWriter(self.board_log_dir)

    def set_train_state(self, state):
        self.train_state = state

    def before_train(self):
        for m in self.metrics:
            m.reset()

    def before_run(self, state):
        ls = [m.tensor for m in self.metrics]
        for i in ls:
            if not (isinstance(i, list) or isinstance(i, tuple)):
                raise ValueError(
                    'metrics should return tuple or list of tensors, got %s' %
                    repr(i))
            for ii in i:
                if not isinstance(ii, str):
                    raise ValueError('metrics tensor be str, got %s' %
                                     repr(ii))
        ls_flt, self.schema = util.flatten(ls)
        #log.debug(ls_flt)
        return ls_flt

    def after_run(self, res_list, state):
        res = util.unflatten(res_list, self.schema)
        for r, m in zip(res, self.metrics):
            m.update(r)

    @property
    def result(self):
        return self._result

    def after_train(self):
        printable = []
        self._result = {}
        for n, m in zip(self.names, self.metrics):
            val = m.eval()
            self._result[n] = val
            assert val.shape == (), 'metrics eval use float'
            printable.append('{}\t{}'.format(n, val))
            self.writer.add_scalar(n, val, self.train_state.gstep)

        if len(printable):
            log.info('*** eval res: %10s ***' % self._name)
            for p in printable:
                log.info(p)
            log.info('******************************')

    def __del__(self):
        if self.writer:
            self.writer.close()


class CheckpointSaverHook(RunHook):
    def __init__(self, saver, per_step=10, skip_step=100):
        self.saver = saver
        self.per_step = per_step
        self.skip_step = skip_step

    def after_run(self, res_list, state):
        if state.gstep % self.per_step == 0 and \
                state.step > self.skip_step:
            self.saver.save(state)
