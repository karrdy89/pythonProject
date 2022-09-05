import dataclasses
import ray
import logger
import logging
import os
import signal
import queue
import subprocess
import time
import traceback
import configparser

from tensorboard import program, default, assets
from ray import exceptions

from utils.resettable_timer import ResettableTimer
from logger import BootLogger


class TensorBoardTool:
    """
    A class that runs and manages the tensorboard service.

    Attributes
    ----------
    _worker : str
        Class name of instance.
    _boot_logger : logger
        A log instance for init process.
    _logger : actor
        The global logger.
    _port : list[int]
        A list of available port.
    _port_use : list[int]
        A List of port in use.
    _tensorboard_thread_queue : queue
        The queue with a tensorboard thread information for expire tensorboard thread.
    _timer : ResettableTimer
        A resettable timer for manage tensorboard thread.
    _before_produce_time : float
        The prior time of tensorboard thread produced for calculate next expire time.
    _TENSORBOARD_PORT_START: int
        Configuration of tensorboard service port range.
    _TENSORBOARD_THREAD_MAX: int
        Configuration of max tensorboard service.
    _EXPIRE_TIME: int
        Configuration of tensorboard service expire time

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    init() -> int:
        Set attributes.
    get_port() -> int:
        Get port number from available port list.
    release_port(int) -> None:
        Release port number from list of port in use.
    expire_tensorboard() -> None:
        Expire tensorboard thread every set time.
    run() -> int:
        Produce tensorboard thread.
    """
    def __init__(self):
        self._worker = type(self).__name__
        self._boot_logger: logger = BootLogger().logger
        self._logger: ray.actor = None
        self._port: list[int] = []
        self._port_use: list[int] = []
        self._tensorboard_thread_queue: queue = queue.Queue()
        self._timer: ResettableTimer = ResettableTimer()
        self._before_produce_time: float = 0
        self._TENSORBOARD_PORT_START: int = 0
        self._TENSORBOARD_THREAD_MAX: int = 0
        self._EXPIRE_TIME: int = 0

    def get_port(self) -> int:
        port = self._port.pop(0)
        self._port_use.append(port)
        return port

    def release_port(self, port: int) -> None:
        self._port_use.remove(port)
        self._port.append(port)

    def expire_tensorboard(self) -> None:
        if self._tensorboard_thread_queue.empty():
            self._timer.stop()
            pass
        tensorboard_thread = self._tensorboard_thread_queue.get(block=True)
        port = tensorboard_thread.port
        tensorboard_info = subprocess.check_output("ps -ef | grep tensorboard", shell=True).decode('utf-8')
        tensorboard_info = tensorboard_info.split("\n")
        try:
            tid = tensorboard_info[0].split()
            tid = tid[1]
            os.kill(int(tid), signal.SIGTERM)
        except Exception as e:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="tensorboard thread not exist: " + str(e))
        else:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="killed: " + str(tid))
            self.release_port(port)
        self._timer.reset(tensorboard_thread.time_diff)

    def init(self) -> int:
        self._logger.info("(" + self._worker + ") " + "init tensorboard service...")

        self._logger.info("(" + self._worker + ") " + "set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._TENSORBOARD_PORT_START = int(config_parser.get("TENSOR_BOARD", "TENSORBOARD_PORT_START"))
            self._TENSORBOARD_THREAD_MAX = int(config_parser.get("TENSOR_BOARD", "TENSORBOARD_THREAD_MAX"))
            self._EXPIRE_TIME = int(config_parser.get("TENSOR_BOARD", "EXPIRE_TIME"))
        except configparser.Error as e:
            self._logger.error("(" + self._worker + ") " + "an error occur when set config...: " + str(e))
            return -1

        self._logger.info("(" + self._worker + ") " + "set global logger...")
        try:
            self._logger = ray.get_actor("logging_service")
        except exceptions as e:
            self._logger.error("(" + self._worker + ") " + "an error occur when set global logger...: " + str(e))

        self._logger.info("(" + self._worker + ") " + "set Tensorboard port range...")
        for i in range(self._TENSORBOARD_THREAD_MAX):
            self._port.append(self._TENSORBOARD_PORT_START + i)
        self._logger.info("(" + self._worker + ") " + "init tensorboard service complete...")
        return 0

    def run(self, dir_path: str) -> int:
        if len(self._port_use) >= self._TENSORBOARD_THREAD_MAX:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="max tensorboard thread exceeded")
            return -1
        port = self.get_port()
        try:
            tensorboard = program.TensorBoard(default.get_plugins(), assets.get_default_assets_zip_provider())
            tensorboard.configure(argv=[None, '--logdir', dir_path, '--port', str(port)])
            tensorboard.launch()
        except Exception as e:
            exc_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="launch tensorboard failed : " + exc_str)
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
                self._timer.set(interval=self._EXPIRE_TIME, function=self.expire_tensorboard)
                self._timer.run()
            return port


@dataclasses.dataclass
class TensorboardThread:
    port: int
    time_diff: float


