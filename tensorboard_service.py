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
import ray
import logging
import os
import signal
import subprocess
import traceback
import configparser

from tensorboard import program, default, assets

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
    _TENSORBOARD_PORT_START: int
        Configuration of tensorboard service port range.
    _TENSORBOARD_THREAD_MAX: int
        Configuration of max tensorboard service.

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
        self._logger: ray.actor.ActorClass | None = None
        self._shared_state: ray.actor.ActorClass | None = None
        self._port: list[int] = []
        self._port_use: list[int] = []
        self._TENSORBOARD_PORT_START: int = 0
        self._TENSORBOARD_THREAD_MAX: int = 0
        self.init()

    def init(self) -> int:
        self._logger = ray.get_actor(Actors.LOGGER)
        self._shared_state = ray.get_actor(Actors.GLOBAL_STATE)
        self._logger.log.remote(level=logging.info, worker=self._worker, msg="init tensorboard service...")
        self._logger.log.remote(level=logging.info, worker=self._worker, msg="set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._TENSORBOARD_PORT_START = int(config_parser.get("TENSOR_BOARD", "TENSORBOARD_PORT_START"))
            self._TENSORBOARD_THREAD_MAX = int(config_parser.get("TENSOR_BOARD", "TENSORBOARD_THREAD_MAX"))
        except configparser.Error as e:
            self._logger.log.remote(level=logging.error, worker=self._worker,
                                    msg="an error occur when set config...: " + str(e))
            return -1

        self._logger.log.remote(level=logging.info, worker=self._worker, msg="set Tensorboard port range...")
        for i in range(self._TENSORBOARD_THREAD_MAX):
            self._port.append(self._TENSORBOARD_PORT_START + i)
        self._logger.log.remote(level=logging.info, worker=self._worker, msg="init tensorboard service complete...")
        return 0

    def get_port(self) -> int | None:
        try:
            port = self._port.pop(0)
        except Exception as exc:
            print(exc.__str__())
            return None
        else:
            self._port_use.append(port)
            return port

    def release_port(self, port: int) -> None:
        self._port_use.remove(port)
        self._port.append(port)

    def expire_tensorboard(self, port, index) -> int:
        # ordered
        tensorboard_info = subprocess.check_output("ps -ef | grep tensorboard", shell=True).decode('utf-8')
        tensorboard_info = tensorboard_info.split("\n")
        tensorboard_info = tensorboard_info[:-3]
        for info in tensorboard_info:
            print(info)
        try:
            tid = tensorboard_info[index].split()
            tid = tid[1]
            print(tid)
            os.kill(int(tid), signal.SIGTERM)
        except Exception as e:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when expire_tensorboard: " + str(e))
            return -1
        else:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="killed: " + str(tid))
            self.release_port(port)
            return 0

    def run(self, dir_path: str) -> int:
        if len(self._port_use) >= self._TENSORBOARD_THREAD_MAX:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="max tensorboard thread exceeded")
            return -2
        port = self.get_port()
        if port is None:
            return -1
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
            return port


