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
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import logging
import six
from time import time

__version__ = '0.1'

log = logging.getLogger(__name__)
try:
    os.mkdir('./log')
except FileExistsError:
    if os.path.isfile('./log'):
        os.remove('./log')

try:
    import tqdm

    class TqdmLoggingHandler(logging.Handler):
        def __init__(self):
            super().__init__()

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    stream_hdl = TqdmLoggingHandler()
except ImportError:
    stream_hdl = logging.StreamHandler()

if os.environ.get('OMPI_COMM_WORLD_RANK') is not None and os.environ.get(
        'OMPI_COMM_WORLD_LOCAL_RANK') is not None:
    rank = os.environ['OMPI_COMM_WORLD_RANK']
    local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    filename = '%d.%s.%s.log' % (int(time()), rank, local_rank)
else:
    filename = '%d.log' % int(time())
file_hdl = logging.FileHandler(os.path.join('./log', filename), mode='w')

formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
)

file_hdl.setFormatter(formatter)

try:
    from colorlog import ColoredFormatter
    fancy_formatter = ColoredFormatter(
        fmt='%(log_color)s[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s'
    )
    stream_hdl.setFormatter(fancy_formatter)
except ImportError:
    stream_hdl.setFormatter(formatter)

log.setLevel(logging.DEBUG)
log.addHandler(stream_hdl)
log.addHandler(file_hdl)
#log.propagate = False

from atarashi.types import *
from atarashi.util import ArgumentParser, parse_hparam, parse_runconfig, parse_file
