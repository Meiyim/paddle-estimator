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

import os
import logging
import six
from time import sleep, time
import multiprocessing

import zmq
""" Never Never Never import paddle.fluid in main process, or any module would import fluid.
"""

log = logging.getLogger(__name__)


def profile(msg):
    def decfn(fn):
        def retfn(*args, **kwargs):
            start = time()
            ret = fn(*args, **kwargs)
            end = time()
            log.debug('%s timecost: %.5f' % (msg, end - start))
            return ret

        return retfn

    return decfn


class Predictor(object):
    def __init__(self, model_dir, device_idx=0):
        import paddle.fluid as F
        log.debug('create predictor on card %d' % device_idx)
        config = F.core.AnalysisConfig(model_dir)
        config.enable_use_gpu(5000, device_idx)
        self._predictor = F.core.create_paddle_predictor(config)

    @profile('paddle')
    def __call__(self, args):
        for i, a in enumerate(args):
            a.name = 'placeholder_%d' % i
        res = self._predictor.run(args)
        return res


def run_worker(model_dir, device_idx, endpoint="ipc://worker.ipc"):
    log.debug("run_worker %s" % device_idx)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv(
        "CUDA_VISIBLE_DEVICES").split(",")[device_idx]
    import paddle.fluid as F
    from propeller.service import interface_pb2
    import propeller.service.utils as serv_utils
    log.debug("import %s" % device_idx)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(endpoint)
    #socket.bind(endpoint)
    try:
        log.debug("Predictor building %s" % device_idx)
        predictor = Predictor(model_dir, 0)
        log.debug("Predictor %s" % device_idx)
    except Exception as e:
        log.exception(e)

    while True:
        #  Wait for next request from client
        try:
            message = socket.recv()
            log.debug("get message %s" % device_idx)
            slots = interface_pb2.Slots()
            slots.ParseFromString(message)
            pts = [serv_utils.slot_to_paddlearray(s) for s in slots.slots]
            ret = predictor(pts)
            slots = interface_pb2.Slots(
                slots=[serv_utils.paddlearray_to_slot(r) for r in ret])
            socket.send(slots.SerializeToString())
        except Exception as e:
            log.exception(e)


class InferencePredictor(object):
    def __init__(self, backend_addr, model_dir, n_devices=1):
        self.backend_addr = backend_addr
        self.model_dir = model_dir
        self.n_devices = n_devices
        self.pool = multiprocessing.Pool(n_devices)

    def start(self):
        for device_idx in range(self.n_devices):
            ret = self.pool.apply_async(run_worker, (
                self.model_dir, device_idx, self.backend_addr))
        return self

    def join(self):
        self.pool.close()
        self.pool.join()


class InferenceProxy(object):
    def listen(self, frontend_addr, backend_addr):
        log.info("InferenceProxy starting...")
        try:
            context = zmq.Context(1)
            # Socket facing clients
            frontend = context.socket(zmq.ROUTER)
            frontend.bind(frontend_addr)
            # Socket facing services
            backend = context.socket(zmq.DEALER)
            backend.bind(backend_addr)
            log.info("Queue init done")
            zmq.device(zmq.QUEUE, frontend, backend)
        except Exception as e:
            log.exception(e)
            log.info("Bringing down zmq device")
        finally:
            frontend.close()
            backend.close()
            context.term()


class InferenceServer(object):
    def __init__(self, model_dir, n_devices):
        self.model_dir = model_dir
        self.n_devices = n_devices

    def listen(self, port):
        frontend_addr = "tcp://*:%s" % port
        backend_addr = "ipc://backend.ipc"
        InferencePredictor(backend_addr, self.model_dir,
                           self.n_devices).start()
        InferenceProxy().listen(frontend_addr, backend_addr)
