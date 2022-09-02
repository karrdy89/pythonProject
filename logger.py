import ray
import logging
import os
from logging.handlers import RotatingFileHandler


@ray.remote
class Logger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.log_base_path = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
        self.init()

    def init(self):
        formatter = logging.Formatter("[%(levelname)s] : %(asctime)s : %(message)s", "%Y-%m-%d %H:%M:%S")
        error_handler = RotatingFileHandler(self.log_base_path+"error.log", mode='a', maxBytes=104857600,
                                            backupCount=100)
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        info_handler = RotatingFileHandler(self.log_base_path+"info.log", mode='a', maxBytes=104857600, backupCount=100)
        info_handler.setFormatter(formatter)
        info_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)

    def log(self, level: int, worker: str, msg: str):
        msg = "(" + worker + ") " + msg
        if level == logging.INFO:
            self.logger.info(msg)
        elif level == logging.ERROR:
            self.logger.error(msg)
        elif level == logging.DEBUG:
            self.logger.debug(msg)


class BootLogger:
    def __init__(self):
        pass