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
from propeller.service.client import InferenceClient

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
    address = "tcp://localhost:5571"
    client = InferenceClient(address)

    data = []
    num = 10
    with open(data_path) as inf:
        for idx, line in enumerate(inf):
            if idx == num:
                break
            np_array = line2nparray(line.strip('\n'))
            data.append(np_array)

    begin = time.time()
    for np_array in data:
        ret = client(*np_array)
    print((time.time() - begin) / num)
