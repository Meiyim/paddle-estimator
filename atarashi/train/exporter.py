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

import sys
import os
import itertools

import numpy as np
import paddle.fluid  as F
import paddle.fluid.layers  as L

from atarashi import log

class Exporter(object):
    def export(self, exe, program, eval_result, state):
        raise NotImplementedError()


class BestExporter(Exporter):
    def __init__(self, export_dir, key, export_highest=True):
        self._export_dir = export_dir
        self._best = None
        if export_highest:
            self.cmp_fn = lambda old, new: old[key] < new[key]
        else:
            self.cmp_fn = lambda old, new: old[key] > new[key]

    def export(self, exe, program, eval_result, state, feeded_var_names, target_vars):
        if self._best is None:
            self._best = eval_result
            log.debug('[Best Exporter]: skip step %d' % state.gstep)
            return
        if self.cmp_fn(self._best, eval_result) :
            log.debug('[Best Exporter]: export to %s' % self._export_dir)
            log.debug(target_vars)
            log.debug(feeded_var_names)
            F.io.save_inference_model(dirname=self._export_dir, feeded_var_names=feeded_var_names, target_vars=target_vars, executor=exe)
            self._best = eval_result
        else:
            log.debug('[Best Exporter]: skip step %s' % state.gstep)

