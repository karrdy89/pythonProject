# *********************************************************************************************************************
# Program Name : logger
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path

import ray
import logging
import configparser
from logging.handlers import TimedRotatingFileHandler
from db.db_util import DBUtil
from statics import ROOT_DIR

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
        logging.basicConfig(level=logging.INFO)
        self._worker: str = type(self).__name__
        self._db: DBUtil | None = None
        self.logger: Logger = logging.getLogger("global")
        self._boot_logger: Logger = BootLogger().logger
        self._log_base_path: str = str(Path(__file__).parent.parent) + "/logs/"
        self._MAX_BACKUP_COUNT: int = 365
        self._MAX_BYTES: int = 104857600

    def init(self) -> int:
        self._boot_logger.info("(" + self._worker + ") " + "init actor : global loger...")
        self._boot_logger.info("(" + self._worker + ") " + "set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read(ROOT_DIR + "/config/config.ini")
            self._MAX_BACKUP_COUNT = int(config_parser.get("LOGGING", "MAX_BACKUP_COUNT"))
            self._MAX_BYTES = int(config_parser.get("LOGGING", "MAX_BYTES"))
        except configparser.Error as e:
            self._boot_logger.error("(" + self._worker + ") " + "an error occur when set config...: " + str(e))
            return -1
        self._boot_logger.info("(" + self._worker + ") " + "set logging parameters...")
        formatter = logging.Formatter("[%(levelname)s] : %(asctime)s : %(message)s", "%Y-%m-%d %H:%M:%S")
        error_handler = TimedRotatingFileHandler(
            self._log_base_path + "{:%Y-%m-%d %H:%M:%S}_error.log".format(datetime.now()),
            when='midnight', interval=1, backupCount=self._MAX_BACKUP_COUNT)
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        info_handler = TimedRotatingFileHandler(
            self._log_base_path + "{:%Y-%m-%d %H:%M:%S}_info.log".format(datetime.now()),
            when='midnight', interval=1, backupCount=self._MAX_BACKUP_COUNT)
        info_handler.setFormatter(formatter)
        info_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        self._db = DBUtil(db_info="MANAGE_DB")
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

    def log_to_db(self, query_name: str, param: dict) -> None:
        self._db.insert(name=query_name, param=param)


class BootLogger:
    """
     A class that provides logger for logging init tasks

     Attributes
     ----------
     logger : Logger
         A Logger class for logging
     _log_base_path : Logger
         The pre-defined Logger class for logging init process.

     Methods
     -------
     __init__():
         Constructs all the necessary attributes.
     """

    def __init__(self):
        logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
        self.logger = logging.getLogger("boot")
        self._log_base_path = str(Path(__file__).parent.parent) + "/logs/"
        formatter = logging.Formatter("[%(levelname)s] : %(asctime)s : %(message)s", "%Y-%m-%d %H:%M:%S")
        log_handler = TimedRotatingFileHandler(
            self._log_base_path + "{:%Y-%m-%d %H:%M:%S}_boot.log".format(datetime.now()),
            when='midnight', interval=1, backupCount=10)
        log_handler.setFormatter(formatter)
        log_handler.setLevel(logging.INFO)
        self.logger.addHandler(log_handler)
