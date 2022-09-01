import dataclasses
import logging
import sys
import os
import signal
import queue
import subprocess
import time
import traceback

from tensorboard import program, default, assets

from utils.resettable_timer import ResettableTimer

# make tensorboard by given name(log path) and port
# return url
# set expire an hour
# - run -> add timer job, activate and reset(queueing the port and next, if same time or less activate if cron???? change cron to time, save time to activate
# get current time and compare??? or calculate interval ) -> how to reset time of scheduler
# (cur - 11:00, interval 1) -> activate on (cur - 12:00, interval ?)
# add tb on (cur - 11:30, interval 1) -> queueing (cur time diff next interval time -> set interval )
# kill in timer and delete in revers proxy
# timer basic unit - sec
# time diff
# 11:10, 11:11, :11:12, 13:00 -> timedif 1,1,1, 1:48, if queue is empty stop and remove

# set config file and read
# run actual model
# handle with db_util
# error handling
# doc string
# test packaging

TENSORBOARD_PORT_START = 6000
TENSORBOARD_THREAD_MAX = 500


class TensorBoardTool:
    def __init__(self):
        self._port: list[int] = []
        self._port_use: list[int] = []
        self._tensorboard_thread_queue: queue = queue.Queue()
        self._timer: ResettableTimer = ResettableTimer()
        self.init()

    def get_port(self):
        port = self._port.pop(0)
        self._port_use.append(port)
        return port

    def release_port(self, port: int):
        self._port_use.remove(port)
        self._port.append(port)

    def expire_tensorboard(self):
        if self._tensorboard_thread_queue.empty():
            self._timer.stop()
            return
        tensorboard_thread = self._tensorboard_thread_queue.get(block=True)
        port = tensorboard_thread.port
        tensorboard_info = subprocess.check_output("ps -ef | grep tensorboard", shell=True).decode('utf-8')
        tensorboard_info = tensorboard_info.split("\n")
        try:
            tid = tensorboard_info[0].split()
            tid = tid[1]
        except IndexError as e:
            print("tensorboard thread not exist: " + str(e))
        else:
            print("killed: " + str(tid))
            os.kill(int(tid), signal.SIGTERM)
            self.release_port(port)
        time_diff = time.time() - tensorboard_thread.produce_time
        print(time_diff)
        self._timer.reset(time_diff)
        return

    def init(self):
        for i in range(TENSORBOARD_THREAD_MAX):
            self._port.append(TENSORBOARD_PORT_START + i)

    def run(self, dir_path: str) -> int:
        port = self.get_port()
        try:
            tensorboard = program.TensorBoard(default.get_plugins(), assets.get_default_assets_zip_provider())
            tensorboard.configure(argv=[None, '--logdir', dir_path, '--port', str(port)])
            url = tensorboard.launch()
            print(url)
        except Exception as e:
            exc_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            print(exc_str)
            print("launch tensorboard failed")
            return -1
        else:
            tensorboard_thread = TensorboardThread(port=port, produce_time=time.time())
            self._tensorboard_thread_queue.put(tensorboard_thread, block=True)
            if not self._timer.is_run:
                self._timer.set(interval=20, function=self.expire_tensorboard)
                self._timer.run()
            return port


@dataclasses.dataclass
class TensorboardThread:
    port: int
    produce_time: float

tb_tool = TensorBoardTool()
tb_tool.run(os.path.dirname(os.path.abspath(__file__))+"/train_logs")

