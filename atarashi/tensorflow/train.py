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
import tensorflow.keras as K

from tensorflow.estimator import ModeKeys, EstimatorSpec

from atarashi.data.functional import Dataset as AtarashiDataset
from atarashi.train.model import Model as AtarashiModel

__all__ = ['train_and_eval', 'predict']


def get_estimator(model_fn_or_model, params, run_config, warm_start_setting):
    def build_estimator(model_fn):
        est_run_config = to_estimator_runconfig(run_config)
        est = tfe.Estimator(
            model_fn=model_fn,
            model_dir=run_config.model_dir,
            config=est_run_config,
            params=params,
            warm_start_from=warm_start_setting)
        return est

    if issubclass(model_fn_or_model, AtarashiModel):

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

        est = build_estimator(model_fn)
    elif inspect.isfunction(model_fn_or_model):
        model_fn = model_fn_or_model
        est = build_estimator(model_fn)
    elif isinstance(model_fn_or_model, K.Model):
        est_run_config = to_estimator_runconfig(run_config)
        est = K.estimator.model_to_estimator(
            keras_model=model_fn_or_model, run_config=est_run_config)
    else:
        raise ValueError('unknown model %s' % model_fn_or_model)
    return est


def train_and_eval(model_class_or_model_fn,
                   params,
                   run_config,
                   train_dataset,
                   eval_dataset=None,
                   warm_start_setting=None,
                   train_hooks=[],
                   eval_hooks=[],
                   exporters=[]):
    est = get_estimator(model_class_or_model_fn, params, run_config,
                        warm_start_setting)
    if not isinstance(eval_da, AtarashiDataset):
        raise ValueError('only accept 1 eval dataset in tensorflor mode')

    if eval_dataset is not None:
        with train_dataset.start(), eval_dataset.start() as train_ds, eval_ds:
            tfe.train_and_eval(
                est,
                tfe.TranSpec(
                    train_ds, max_steps=run_config.max_steps),
                tfe.EvalSpec(
                    eval_ds, steps=None))
    else:
        est.train(train_ds, max_steps=run_config.max_steps)


def predict(model_class_or_model_fn,
            params,
            model_dir,
            infer_dataset,
            run_config=None,
            steps=-1,
            split_batch=True):
    pass
