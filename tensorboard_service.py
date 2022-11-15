# *********************************************************************************************************************
# Program Name : tensorboard_service
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import dataclasses
import ray
import logger
import logging
import os
import signal
import queue
import subprocess
import traceback
import configparser
from datetime import datetime, timedelta

from tensorboard import program, default, assets

from utils.resettable_timer import ResettableTimer
from apscheduler.schedulers.background import BackgroundScheduler

from statics import Actors


class TensorBoardTool:
    """
    A class that runs and manages the tensorboard service.

    Attributes
    ----------
    _worker : str
        Class name of instance.
    _logger : actor
        The global logger.
    _port : list[int]
        A list of available port.
    _port_use : list[int]
        A List of port in use.
    _tensorboard_thread_queue : queue
        The queue with a tensorboard thread information for expire tensorboard thread.
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
        self._logger: ray.actor = None
        self._port: list[int] = []
        self._port_use: list[int] = []
        self._tensorboard_thread_queue: queue = queue.Queue()
        self._before_produce_time: float = 0
        self._TENSORBOARD_PORT_START: int = 0
        self._TENSORBOARD_THREAD_MAX: int = 0
        self._EXPIRE_TIME: int = 0
        self._scheduler = BackgroundScheduler()
        self._scheduler.start()
        self.init()

    def init(self) -> int:
        self._logger = ray.get_actor(Actors.LOGGER)
        self._logger.log.remote(level=logging.info, worker=self._worker, msg="init tensorboard service...")
        self._logger.log.remote(level=logging.info, worker=self._worker, msg="set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._TENSORBOARD_PORT_START = int(config_parser.get("TENSOR_BOARD", "TENSORBOARD_PORT_START"))
            self._TENSORBOARD_THREAD_MAX = int(config_parser.get("TENSOR_BOARD", "TENSORBOARD_THREAD_MAX"))
            self._EXPIRE_TIME = int(config_parser.get("TENSOR_BOARD", "EXPIRE_TIME"))
        except configparser.Error as e:
            self._logger.log.remote(level=logging.error, worker=self._worker,
                                    msg="an error occur when set config...: " + str(e))
            return -1

        self._logger.log.remote(level=logging.info, worker=self._worker, msg="set Tensorboard port range...")
        for i in range(self._TENSORBOARD_THREAD_MAX):
            self._port.append(self._TENSORBOARD_PORT_START + i)
        self._logger.log.remote(level=logging.info, worker=self._worker, msg="init tensorboard service complete...")
        return 0

    def get_port(self) -> int:
        try:
            port = self._port.pop(0)
        except Exception:
            pass
        else:
            self._port_use.append(port)
            return port

    def release_port(self, port: int) -> None:
        self._port_use.remove(port)
        self._port.append(port)

    def expire_tensorboard(self) -> None:
        if self._tensorboard_thread_queue.empty():
            return
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
            print(e)
            exc_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="launch tensorboard failed : " + exc_str)
            self.release_port(port)
            return -1
        else:
            tensorboard_thread = TensorboardThread(port=port)
            self._tensorboard_thread_queue.put(tensorboard_thread, block=True)
            run_date = datetime.now() + timedelta(seconds=self._EXPIRE_TIME)
            self._scheduler.add_job(self.expire_tensorboard, "date", run_date=run_date)
            return port


@dataclasses.dataclass
class TensorboardThread:
    port: int


