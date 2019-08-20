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
    def __init__(self, model_dir, did):
        import paddle.fluid as F
        log.debug('create predictor on card %d' % did)
        config = F.core.AnalysisConfig(model_dir)
        config.enable_use_gpu(5000, did)
        self._predictor = F.core.create_paddle_predictor(config)

    @profile('paddle')
    def __call__(self, args):
        for i, a in enumerate(args):
            a.name = 'placeholder_%d' % i
        res = self._predictor.run(args)
        return res


def process(model_dir, did):
    #log.info("start process %s" % did)
    print("start process %s" % did)
    import paddle.fluid as F
    print("import fluid done")
    from wave.service import interface_pb2
    print("import pb2 done")
    import wave.paddle.service.utils as serv_utils
    print("import service done")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect("ipc://worker.ipc")
    #socket.bind("tcp://*:5571")
    print("init predictor")
    predictor = Predictor(model_dir, did)
    print("init predictor done")
    while True:
        #  Wait for next request from client
        message = socket.recv()
        slots = interface_pb2.Slots()
        slots.ParseFromString(message)
        pts = [serv_utils.slot_to_paddlearray(s) for s in slots.slots]
        ret = predictor(pts)
        slots = interface_pb2.Slots(
            slots=[serv_utils.paddlearray_to_slot(r) for r in ret])
        socket.send(slots.SerializeToString())


def main():
    model_dir = "/home/work/suweiyue/Release/infer_xnli/model/"
    #model_dir = '/home/work/chenxuyi/playground/grpc_play/ernie2.0/',
    num = 2
    #process(model_dir, 0)
    #return 
    pool = multiprocessing.Pool(num)
    for i in range(num):
        ret = pool.apply_async(process, (model_dir, i))
    print("start proxy")
    try:
        context = zmq.Context(1)
        # Socket facing clients
        frontend = context.socket(zmq.ROUTER)
        frontend.bind("tcp://*:5571")
        # Socket facing services
        backend = context.socket(zmq.DEALER)
        backend.bind("ipc://worker.ipc")
        print("queue done")
        zmq.device(zmq.QUEUE, frontend, backend)
    except Exception as e:
        print(e)
        print("bringing down zmq device")
    finally:
        frontend.close()
        backend.close()
        context.term()


if __name__ == "__main__":
    main()

# def serve(model_dir, host, num_concurrent=None):
# #if six.PY2:
# #raise RuntimeError('wave service work in python3 only')
# num_worker = len(F.cuda_places(
# )) if num_concurrent is None else num_concurrent
# #pool = ThreadPoolExecutor(num_worker)

# class Predictor(object):
# def __init__(self, did):
# log.debug('create predictor on card %d' % did)
# config = F.core.AnalysisConfig(model_dir)
# config.enable_use_gpu(5000, did)
# self._predictor = F.core.create_paddle_predictor(config)

# @profile('paddle')
# def __call__(self, args):
# for i, a in enumerate(args):
# a.name = 'placeholder_%d' % i
# res = self._predictor.run(args)
# return res

# predictor_context = {}

# class InferenceService(interface_pb2_grpc.InferenceServicer):
# @profile('service')
# def Infer(self, request, context):
# try:
# slots = request.slots
# current_thread = threading.current_thread()
# log.debug('%d slots received dispatch to thread %s' %
# (len(slots), current_thread))
# if current_thread not in predictor_context:
# did = list(pool._threads).index(current_thread)
# log.debug('spawning worker thread %d' % did)
# predictor = Predictor(did)
# predictor_context[current_thread] = predictor
# else:
# predictor = predictor_context[current_thread]
# slots = [serv_utils.slot_to_paddlearray(s) for s in slots]
# ret = predictor(slots)
# response = [serv_utils.paddlearray_to_slot(r) for r in ret]
# except Exception as e:
# log.exception(e)
# raise e
# return interface_pb2.Slots(slots=response)

# server = grpc.server(pool)
# interface_pb2_grpc.add_InferenceServicer_to_server(InferenceService(),
# server)
# server.add_insecure_port(host)
# server.start()
# log.info('server started on %s...' % host)
# try:
# while True:
# sleep(100000)
# except KeyboardInterrupt as e:
# pass
# log.info('server stoped...')

# if __name__ == '__main__':
# from wave import log
# log.setLevel(logging.DEBUG)
# serve(
# '/home/work/chenxuyi/playground/grpc_play/ernie2.0/',
# '10.255.138.19:8334',
# num_concurrent=3)
