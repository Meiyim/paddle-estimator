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

import zmq
import numpy as np

import propeller.service.utils as serv_utils


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
        request = serv_utils.nparray_list_serialize(args)
        self.socket.send(request)
        reply = self.socket.recv()
        ret = serv_utils.nparray_list_deserialize(reply)
        return ret
