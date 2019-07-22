# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import struct
import logging
import argparse
import numpy as np
import collections
from distutils import dir_util

#from utils import print_arguments 
import paddle.fluid as F
from paddle.fluid.proto import framework_pb2

log = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
)
console = logging.StreamHandler()
console.setFormatter(formatter)
log.addHandler(console)
log.setLevel(logging.DEBUG)


def parse(filename):
    with open(filename, 'rb') as f:
        read = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        _, = read('I')  # version
        log.debug(_)
        lodsize, = read('Q')
        log.debug(lodsize)
        if lodsize != 0:
            raise RuntimeError('shit, it is LOD tensor!!! abort')
        _, = read('I')  # version
        log.debug(_)
        pbsize, = read('i')
        log.debug(pbsize)
        data = f.read(pbsize)
        proto = framework_pb2.VarType.TensorDesc()
        proto.ParseFromString(data)

        log.info('type: [%s] dim %s' % (proto.data_type, proto.dims))
        if proto.data_type == framework_pb2.VarType.FP32:
            T = 'f'
            data = f.read()
            num = len(data) // struct.calcsize(T)
            floats = struct.unpack('%d%s' % (num, T), data)
            arr = np.array(floats, dtype=np.float32).reshape(proto.dims)
        else:
            raise RuntimeError('Unknown dtype %s' % proto.data_type)

        return arr


def show(arr):
    print(repr(arr))


def dump(arr, path):
    path = os.path.join(args.to, path)
    path = '%s.np' % path
    log.info('dump to %s' % path)
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass
    arr.dump(open(path, 'wb'))


def list_dir(dir_or_file):
    if os.path.isfile(dir_or_file):
        return [dir_or_file]
    else:
        return [
            os.path.join(i, kk) for i, _, k in os.walk(dir_or_file) for kk in k
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['show', 'dump'], type=str)
    parser.add_argument('file_or_dir', type=str)
    parser.add_argument('-t', "--to", type=str, default=None)
    parser.add_argument('-v', "--verbose", action='store_true')
    args = parser.parse_args()

    files = list_dir(args.file_or_dir)
    parsed_arr = map(parse, files)
    if args.mode == 'show':
        for arr in parsed_arr:
            show(arr)
    elif args.mode == 'dump':
        if args.to is None:
            raise ValueError('--to dir_name not specified')
        for arr, path in zip(parsed_arr, files):
            dump(arr, path)
