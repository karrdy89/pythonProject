import ray
import logging
import os
import configparser
from logging.handlers import RotatingFileHandler


@ray.remote
class Logger:
    """
    A ray actor class for global logging task

    Attributes
    ----------
    _worker : str
        The class name of instance.
    logger : Logger
        A Logger class for logging
    _boot_logger : Logger
        The pre-defined Logger class for logging init process.
    _log_base_path : str
        The path of log directory.
    _MAX_BACKUP_COUNT : int
        Configuration of max backup log file number.
    _MAX_BYTES : int
        Configuration of max bytes of log file.

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    init(self) -> int
        Set attributes.
    log(self, level: int, worker: str, msg: str) -> None:
        Logging given data to files
    """
    def __init__(self):
        self._worker: str = type(self).__name__
        self.logger: Logger = logging.getLogger()
        self._boot_logger: Logger = BootLogger().logger
        self._log_base_path: str = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
        self._MAX_BACKUP_COUNT: int = 100
        self._MAX_BYTES: int = 104857600

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
        error_handler = RotatingFileHandler(self._log_base_path + "error.log", mode='a', maxBytes=self._MAX_BYTES,
                                            backupCount=self._MAX_BACKUP_COUNT)
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        info_handler = RotatingFileHandler(self._log_base_path + "info.log", mode='a', maxBytes=self._MAX_BYTES,
                                           backupCount=self._MAX_BACKUP_COUNT)
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
    """
     A class that provides logger for logging init tasks

     Attributes
     ----------
     logger : Logger
         A Logger class for logging
     log_base_path : Logger
         The pre-defined Logger class for logging init process.

     Methods
     -------
     __init__():
         Constructs all the necessary attributes.
     """
    def __init__(self):
        self.logger = logging.getLogger()
        self.log_base_path = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
        formatter = logging.Formatter("[%(levelname)s] : %(asctime)s : %(message)s", "%Y-%m-%d %H:%M:%S")
        log_handler = RotatingFileHandler(self.log_base_path + "error.log", mode='a', maxBytes=104857600,
                                          backupCount=5)
        log_handler.setFormatter(formatter)
        log_handler.setLevel(logging.INFO)
        self.logger.addHandler(log_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
