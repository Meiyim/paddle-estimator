#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# File: paddle/service/__main__.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2019/08/14 17:01:17
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import logging
import six
import asyncio
import threading
from collections import defaultdict

import grpc
from atarashi.paddle.service import interface_pb2
from atarashi.paddle.service import interface_pb2_grpc
import atarashi.paddle.service.utils as serv_utils

from concurrent.futures import ThreadPoolExecutor

import paddle.fluid as F
import threading

from time import sleep, time

log = logging.getLogger(__name__)


def serve(model_dir, host, num_concurrent=None):
    class Predictor(object):
        def __init__(self):
            pass

        def __call__(self, args):
            return [arg + 1 for arg in args]

    predictor_context = defaultdict(Predictor)

    class InferenceService(interface_pb2_grpc.InferenceServicer):
        def Infer(self, request, context):
            try:
                slots = request.slots
                tid = threading.get_ident()
                log.info('slots received %s dispatch to %s' % (slots, tid))
                ctx = predictor_context[tid]
                slots = [serv_utils.slot_to_numpy(s) for s in slots]
                ret = ctx(slots)
                response = [serv_utils.numpy_to_slot(r) for r in ret]
            except Exception as e:
                log.exception(e)
                raise e
            return interface_pb2.Slots(slots=response)

    num_worker = len(F.cuda_places(
    )) if num_concurrent is None else num_concurrent
    pool = ThreadPoolExecutor(num_worker)
    server = grpc.server(pool)
    interface_pb2_grpc.add_InferenceServicer_to_server(InferenceService(),
                                                       server)
    server.add_insecure_port(host)
    server.start()
    log.info('server started on %s...' % host)
    try:
        while True:
            sleep(100000)
    except KeyboardInterrupt as e:
        pass
    log.info('server stoped...')


if __name__ == '__main__':
    from atarashi import log
    if six.PY2:
        raise RuntimeError('atarashi service work in python3 only')
    serve('./', '10.255.138.19:8334', 3)
