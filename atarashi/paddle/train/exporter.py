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
import os
import itertools
import six
import abc
import logging

import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L

from atarashi.paddle.train import Saver
from atarashi.types import InferenceSpec

log = logging.getLogger(__name__)


@six.add_metaclass(abc.ABCMeta)
class Exporter():
    @abc.abstractmethod
    def export(self, exe, program, eval_result, state):
        raise NotImplementedError()


class BestExporter(Exporter):
    def __init__(self, export_dir, cmp_fn):
        self._export_dir = export_dir
        self._best = None
        self.cmp_fn = cmp_fn

    def export(self, exe, program, eval_model_spec, eval_result, state):
        log.debug('New evaluate result: %s \nold: %s' %
                  (repr(eval_result), repr(self._best)))
        if self._best is None:
            self._best = eval_result
            log.debug('[Best Exporter]: skip step %d' % state.gstep)
            return
        if self.cmp_fn(old=self._best, new=eval_result):
            log.debug('[Best Exporter]: export to %s' % self._export_dir)
            if eval_model_spec.inference_spec is not None:
                inf_sepc_dict = eval_model_spec.inference_spec
                if not isinstance(inf_sepc_dict, dict):
                    inf_sepc_dict = {'inference': inf_sepc_dict}
                for inf_sepc_name, inf_sepc in six.iteritems(inf_sepc_dict):
                    if not isinstance(inf_sepc, InferenceSpec):
                        raise ValueError('unkonw inference spec type: %s' % v)

                    save_dir = os.path.join(self._export_dir, inf_sepc_name)
                    log.debug('[Best Exporter]: save inference model: %s to %s'
                              % (inf_sepc_name, save_dir))
                    feed_var = [i.name for i in inf_sepc.inputs]
                    fetch_var = inf_sepc.outputs
                    F.io.save_inference_model(
                        save_dir,
                        feed_var,
                        fetch_var,
                        exe,
                        main_program=program)
            else:
                log.debug('[Best Exporter]: save regular model')
                saver = Saver(
                    self._export_dir, exe, program=program, max_ckpt_to_keep=1)
                saver.save(state)
            self._best = eval_result
        else:
            log.debug('[Best Exporter]: skip step %s' % state.gstep)
