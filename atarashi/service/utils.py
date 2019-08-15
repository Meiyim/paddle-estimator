#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from atarashi.service import interface_pb2
from atarashi.service import interface_pb2_grpc


def slot_to_numpy(slot):
    if slot.type == interface_pb2.Slot.FP32:
        dtype = np.float32
        type_str = 'f'
    elif slot.type == interface_pb2.Slot.INT32:
        type_str = 'i'
        dtype = np.int32
    elif slot.type == interface_pb2.Slot.INT64:
        dtype = np.int64
        type_str = 'q'
    else:
        raise RuntimeError('know type %s' % slot.type)
    num = len(slot.data) // struct.calcsize(type_str)
    arr = struct.unpack('%d%s' % (num, type_str), slot.data)
    shape = slot.dims
    ret = np.array(arr, dtype=dtype).reshape(shape)
    return ret


def numpy_to_slot(arr):
    if arr.dtype == np.float32:
        dtype = interface_pb2.Slot.FP32
    elif arr.dtype == np.int32:
        dtype = interface_pb2.Slot.INT32
    elif arr.dtype == np.int64:
        dtype = interface_pb2.Slot.INT64
    else:
        raise RuntimeError('know type %s' % arr.dtype)
    pb = interface_pb2.Slot(
        type=dtype, dims=list(arr.shape), data=arr.tobytes())
    return pb
