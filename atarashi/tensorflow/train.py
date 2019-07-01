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
import tensorflow as tf
from tensorflow.estimator import ModeKeys, EstimatorSpec

from .model import Model

__all__ = ['train_and_eval', 'predict']


def get_model_fn(model_fn_or_model, features, mode, params, run_config):
    if issubclass(model_fn_or_model, train.Model):

        def model_fn(features, labels, mode, params, run_config):
            if mode != RunMode.PREDICT:
                fea, label = features[:-1], features[-1]
            else:
                fea = features

            model = model_fn_or_model(params, mode, run_config=run_config)
            pred = model.forward(fea)

            if mode == ModeKeys.TRAIN:
                loss = model.loss(pred, label)
                model.backward(loss)
                return EstimatorSpec(loss=loss, predictions=pred, mode=mode)
            elif mode == ModeKeys.EVAL:
                loss = model.loss(pred, label)
                me = model.metrics(pred, label)
                if 'loss' not in me:
                    me['loss'] = metrics.Mean(loss)
                return EstimatorSpec(
                    loss=loss, predictions=pred, metrics=me, mode=mode)
            elif mode == ModeKeys.PREDICT:
                return EstimatorSpec(predictions=pred, mode=mode)
            else:
                raise RuntimeError('unknown run mode %s' % mode)
    elif inspect.isfunction(model_fn_or_model):
        model_fn = model_fn_or_model
    else:
        raise ValueError('unknown model %s' % model_fn_or_model)
    return model_fn


def train_and_eval(model_class_or_model_fn,
                   params,
                   run_config,
                   train_dataset,
                   eval_dataset=None,
                   warm_start_setting=None,
                   train_hooks=[],
                   eval_hooks=[],
                   exporters=[]):
    model_fn = get_model_fn(model_class_or_model_fn)


def predict(model_class_or_model_fn,
            params,
            model_dir,
            infer_dataset,
            run_config=None,
            steps=-1,
            split_batch=True):
    pass
