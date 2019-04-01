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
import logging
import os
import itertools
import json
import numpy as np

class Model(object):
    def __init__(self, config, mode):
        """
        Args:
            config (dict): hyper param
            mode (atarashi.RunMode):  will creat `TRAIN` and `EVAL` model in atarashi.train_and_eval
        """
        self.mode = mode

    def forward(self, features):
        """
        Args:
            features (list of Tensor): depends on your Dataset.output_shapes
        Returns:
            return (Tensor): 
        """
        pass


    def loss(self, predictions, label):
        """
        Args:
            predictions (Tensor): result of  `self.forward`
            label (Tensor): depends on your Dataset.output_shapes
        Returns:
            return (paddle scalar): loss
        

        """
        pass


    def backward(self, loss):
        """
        Call in TRAIN mode
        Args:
            loss (Tensor): result of `self.loss`
        Returns:
            None
        """
        pass

        
    def metrics(self, predictions, label):
        """
        Call in EVAL mode
        Args:
            predictions (Tensor): result of  `self.forward`
            label (Tensor): depends on your Dataset.output_shapes
        Returns:
            (dict): k-v map like: {"metrics_name": atarashi.Metrics } 
        """
        return {}


