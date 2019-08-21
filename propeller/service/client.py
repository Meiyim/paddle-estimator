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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import struct

import grpc
from propeller.service import interface_pb2
from propeller.service import interface_pb2_grpc
import propeller.service.utils as serv_utils

from time import time


class InferenceClient(object):
    def __init__(self, host):
        conn = grpc.insecure_channel(host)
        print('try connect to host %s' % host)
        self.client = interface_pb2_grpc.InferenceStub(channel=conn)

    def __call__(self, *args):
        for arg in args:
            if not isinstance(arg, np.ndarray):
                raise ValueError('expect ndarray slot data, got %s' %
                                 repr(arg))
        args = [serv_utils.numpy_to_slot(arg) for arg in args]
        slots = interface_pb2.Slots(slots=args)
        response = self.client.Infer(slots)
        ret = [serv_utils.slot_to_numpy(r) for r in response.slots]
        return ret
