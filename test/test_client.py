#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# File: test_client.py
# Author: suweiyue(suweiyue@baidu.com)
# Date: 2019/08/21 23:34:25
#
########################################################################
"""
    Comment.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import time

import numpy as np
from wave.service.client import InferenceClient

if __name__ == "__main__":

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

    data_path = "/home/work/suweiyue/Release/infer_xnli/seq128_data/dev_ds"
    line = open(data_path).readline().strip('\n')
    np_array = line2nparray(line)
    num = 10
    begin = time.time()
    client = InferenceClient()
    for request in range(num):
        print(request)
        ret = client(*np_array)
        #print(ret)
    print((time.time() - begin) / num)
