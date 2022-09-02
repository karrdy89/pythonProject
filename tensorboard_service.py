import dataclasses
import os
import signal
import queue
import subprocess
import time
import traceback

from tensorboard import program, default, assets

from utils.resettable_timer import ResettableTimer


TENSORBOARD_PORT_START = 6000
TENSORBOARD_THREAD_MAX = 500
DEFAULT_EXPIRE_TIME = 3600


class TensorBoardTool:
    def __init__(self):
        self._worker = type(self).__name__
        self._port: list[int] = []
        self._port_use: list[int] = []
        self._tensorboard_thread_queue: queue = queue.Queue()
        self._timer: ResettableTimer = ResettableTimer()
        self._before_produce_time: float = 0
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
        self._timer.reset(tensorboard_thread.time_diff)
        return

    def init(self):
        for i in range(TENSORBOARD_THREAD_MAX):
            self._port.append(TENSORBOARD_PORT_START + i)

    def run(self, dir_path: str) -> int:
        if len(self._port_use) >= TENSORBOARD_THREAD_MAX:
            print("max tensorboard thread exceeded")
            return -1
        port = self.get_port()
        try:
            tensorboard = program.TensorBoard(default.get_plugins(), assets.get_default_assets_zip_provider())
            tensorboard.configure(argv=[None, '--logdir', dir_path, '--port', str(port)])
            tensorboard.launch()
        except Exception as e:
            exc_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            print(exc_str)
            print("launch tensorboard failed")
            return -1
        else:
            cur_time = time.time()
            time_diff = 0
            if self._before_produce_time != 0:
                time_diff = cur_time - self._before_produce_time
            self._before_produce_time = cur_time
            tensorboard_thread = TensorboardThread(port=port, time_diff=time_diff)
            self._tensorboard_thread_queue.put(tensorboard_thread, block=True)
            if not self._timer.is_run:
                self._timer.set(interval=DEFAULT_EXPIRE_TIME, function=self.expire_tensorboard)
                self._timer.run()
            return port


@dataclasses.dataclass
class TensorboardThread:
    port: int
    time_diff: float


