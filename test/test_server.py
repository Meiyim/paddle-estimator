#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# File: test_server.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2019/08/21 21:17:24
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
from propeller.service.server import InferenceServer
from propeller.service.server import run_worker

if __name__ == "__main__":
    frontend_addr = "tcp://*:5571"
    model_dir = "/home/work/suweiyue/Release/infer_xnli/model/"
    n_devices = len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))
    InferenceServer(frontend_addr, model_dir, n_devices)
    #run_worker(model_dir, 0, frontend_addr)
