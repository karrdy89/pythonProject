import ray
import logging
import os
import configparser
from logging.handlers import RotatingFileHandler


@ray.remote
class Logger:
    """
    A ray actor class for global logging

    Attributes
    ----------
    _worker : str
        Class name of instance.
    logger : Logger
        A name of pipeline. (model_name:version)
    _boot_logger : Logger
        The list of sequence name that defined in pipeline.yaml.
    _log_base_path : str
        The path to pipeline definition file.
    _MAX_BACKUP_COUNT : int
        The actor handel of global logger.
    _MAX_BYTES : int
        The actor handel of global data store.

    Methods
    -------
    __init__():
        Constructs all the necessary attributes for the person object.
    _get_piepline_definition() -> dict:
        Create dictionary from pipeline.yaml.
    run_pipeline(name: str, version: str, train_info: TrainInfo) -> dict:
        Set pipline attributes and run pipeline.
    trigger_pipeline(train_info) -> int:
        Run each component of piepline
    on_pipeline_end() -> None:
        Ask shared_state actor to kill this pipeline
    """
    def __init__(self):
        self._worker: str = type(self).__name__
        self.logger: Logger = logging.getLogger()
        self._boot_logger: Logger = BootLogger().logger
        self._log_base_path: str = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
        self._MAX_BACKUP_COUNT: int = 100
        self._MAX_BYTES: int = 104857600
        self.init()

    def init(self) -> int:
        self._boot_logger.info("(" + self._worker + ") " + "init actor : global loger...")
        self._boot_logger.info("(" + self._worker + ") " + "set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._MAX_BACKUP_COUNT = int(config_parser.get("LOGGING", "MAX_BACKUP_COUNT"))
            self._MAX_BYTES = int(config_parser.get("LOGGING", "MAX_BYTES"))
        except configparser.Error as e:
            self._boot_logger.error("(" + self._worker + ") " + "an error occur when set config...: " + str(e))
            return -1
        self._boot_logger.info("(" + self._worker + ") " + "set logging parameters...")
        formatter = logging.Formatter("[%(levelname)s] : %(asctime)s : %(message)s", "%Y-%m-%d %H:%M:%S")
        error_handler = RotatingFileHandler(self._log_base_path + "error.log", mode='a', maxBytes=self._max_bytes,
                                            backupCount=self._max_backup_count)
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        info_handler = RotatingFileHandler(self._log_base_path + "info.log", mode='a', maxBytes=self._max_bytes,
                                           backupCount=self._max_backup_count)
        info_handler.setFormatter(formatter)
        info_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        self._boot_logger.info("(" + self._worker + ") " + "init actor complete: global loger...")
        return 0

    def log(self, level: int, worker: str, msg: str) -> None:
        msg = "(" + worker + ") " + msg
        if level == logging.INFO:
            self.logger.info(msg)
        elif level == logging.ERROR:
            self.logger.error(msg)
        elif level == logging.DEBUG:
            self.logger.debug(msg)


class BootLogger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.log_base_path = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
        formatter = logging.Formatter("[%(levelname)s] : %(asctime)s : %(message)s", "%Y-%m-%d %H:%M:%S")
        log_handler = RotatingFileHandler(self.log_base_path + "error.log", mode='a', maxBytes=104857600,
                                          backupCount=5)
        log_handler.setFormatter(formatter)
        log_handler.setLevel(logging.INFO)
        self.logger.addHandler(log_handler)
