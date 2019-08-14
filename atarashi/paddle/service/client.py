#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# File: paddle/client.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2019/08/14 20:05:25
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

import grpc
from atarashi.paddle.service import interface_pb2
from atarashi.paddle.service import interface_pb2_grpc
import atarashi.paddle.service.utils as serv_utils

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
