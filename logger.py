import ray
import logging
import os
from logging.handlers import TimedRotatingFileHandler


class Logger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.log_base_path = os.path.dirname(os.path.abspath(__file__)) + "/logs/"
        self._on_load()

    def _on_load(self):
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S")
        error_handler = TimedRotatingFileHandler(self.log_base_path+"error.log", when="h", interval=1, backupCount=720)
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        info_handler = TimedRotatingFileHandler(self.log_base_path+"info.log", when="h", interval=1, backupCount=720)
        info_handler.setFormatter(formatter)
        info_handler.setLevel(logging.INFO)
        debug_handler = TimedRotatingFileHandler(self.log_base_path+"debug.log", when="h", interval=1, backupCount=720)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(debug_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)

    def log(self, level: int, msg: str):
        if level == logging.INFO:
            self.logger.info(msg)
