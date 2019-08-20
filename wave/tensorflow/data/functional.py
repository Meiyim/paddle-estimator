# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

import paddle.fluid as F
import paddle.fluid.layers as L
import tensorflow.data as tfd

from wave.data.functional import Dataset as DatasetBase


class Dataset(DatasetBase):
    def placeholders(self):
        raise NotImplementedError()

    def features(self):
        raise NotImplementedError()

    def _infer_shapes_and_types(self):
        if self.generator is not None and self.name is not None:
            log.info('Try to infer data shapes & types from generator')
            first_value = next(self.generator())
            shapes, types = [], []
            for v in first_value:
                if not isinstance(v, np.ndarray):
                    raise ValueError(
                        'dataset generator should use numpy elements, got %s' %
                        first_value)
                shapes.append(v.shape)
                types.append(v.dtype.name)
            self._data_shapes = shapes
            self._data_types = types
            log.info('Dataset `%s` has data_shapes: %s data_types: %s' %
                     (self.name, repr(shapes), repr(types)))
        else:
            raise ValueError(
                'Try to infer data shapes or types from incomplete Dataset')

    def start(self):
        def gen():
            try:
                for i in self.generator():
                    yield i
            except Exception as e:
                log.exception(e)

        return TFreaderContext(gen)


class TFreaderContext(object):
    def __init__(self, generator):
        self._gen = generator

    def __enter__(self):
        shapes = tuple(self.data_shapes)
        types = tuple(self.data_types)
        ds = lambda: tfd.Dataset.from_generator_func(self._gen, shapes, types)
        return ds

    def __exit__(self, err_type, err_value, trace):
        pass
