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

import sys
import random
import time
import struct

import zmq
import numpy as np
from wave.service import interface_pb2
import wave.service.utils as serv_utils


class InferenceClient(object):
    def __init__(self):
        port = "5571"
        context = zmq.Context()
        print("Connecting to server...")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:%s" % port)

    def __call__(self, *args):
        for arg in args:
            if not isinstance(arg, np.ndarray):
                raise ValueError('expect ndarray slot data, got %s' %
                                 repr(arg))

        args = [serv_utils.numpy_to_slot(arg) for arg in args]
        slots = interface_pb2.Slots(slots=args)
        self.socket.send(slots.SerializeToString())
        message = self.socket.recv()
        slots = interface_pb2.Slots()
        slots.ParseFromString(message)
        ret = [serv_utils.slot_to_numpy(r) for r in slots.slots]
        return ret


def line2nparray(line):
    slots = [slot.split(':') for slot in line.split(';')]
    dtypes = ["int64", "int64", "int64", "float32"]
    data_list = [
        np.reshape(
            np.array(
                [float(num) for num in data.split(" ")], dtype=dtype),
            [int(s) for s in shape.split(" ")])
        for (shape, data), dtype in zip(slots, dtypes)
    ]
    return data_list


if __name__ == "__main__":
    data_path = "/home/work/suweiyue/Release/infer_xnli/seq128_data/dev_ds"
    line = open(data_path).readline().strip('\n')
    np_array = line2nparray(line)
    num = 100
    begin = time.time()
    client = InferenceClient()
    for request in range(num):
        #print(request)
        ret = client(*np_array)
        #print(ret)
    print((time.time() - begin) / num)
