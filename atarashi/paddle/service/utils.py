#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# File: utils.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2019/08/14 20:36:55
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import struct

from atarashi.paddle.service import interface_pb2
from atarashi.paddle.service import interface_pb2_grpc


def slot_to_numpy(slot):
    if slot.type == interface_pb2.Slot.FP32:
        dtype = np.float32
        type_str = 'f'
    elif slot.type == interface_pb2.Slot.INT32:
        type_str = 'i'
        dtype = np.int32
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
    if arr.dtype == np.int32:
        dtype = interface_pb2.Slot.INT32
    else:
        raise RuntimeError('know type %s' % arr.dtype)
    pb = interface_pb2.Slot(
        type=dtype, dims=list(arr.shape), data=arr.tobytes())
    return pb
