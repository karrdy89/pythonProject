import configparser
import logging
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import ray
from ray import actor

from pipeline import TrainResult
from logger import BootLogger
from statics import Actors


@ray.remote
class SharedState:
    """
    A ray actor class to serve and inference tensorflow model

    Attributes
    ----------
    _worker : str
        The class name of instance.
    _logger : actor
        A Logger class for logging
    _boot_logger : Logger
        The pre-defined Logger class for logging init process.
    _actors : OrderedDict[str, actor]
        An OrderedDict of actor handles (Ex. {actor_name : ray.actor})
    _pipline_result : OrderedDict[str, dict]
        An OrderedDict of current pipeline progress (Ex. {pipeline_name: {task_1: state, task_2: state ,...} ,...})
    _train_result : OrderedDict[str, TrainResult]
        An OrderedDict of current training result (Ex. {model1_version: {metric_1: n, metric_2: k, ...}, ...}
    _PIPELINE_MAX : int
        Configuration that number of max concurrent pipeline executions

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    init() -> int
        Set attributes.
    set_actor(name: str, act: actor) -> None | int:
        Store actor handel to _actors with given actor name and handle
    is_actor_exist(name) -> bool:
        Return true / false if given actor name exists in _actors
    delete_actor(name: str) -> None:
        Delete actor data from _actors with given name
    kill_actor(name: str) -> int:
        Kill running actor with given name
    set_pipeline_result(name: str, pipe_result: dict) -> None:
        Store pipeline result to _pipline_result with given data
    delete_pipeline_result(name: str) -> None:
        Delete pipeline result data from _pipline_result with given name
    get_pipeline_result(name: str) -> dict:
        Return pipeline result data from _pipline_result with given name
    set_train_result(name: str, train_result: TrainResult) -> None:
        Store train result to _train_result with given data
    get_train_result(name: str) -> dict:
        Return train result data from _train_result with given name
    delete_train_result(name: str) -> None:
        Delete train result data from _train_result with given name
    """
    def __init__(self):
        self._worker = type(self).__name__
        self._logger: actor = None
        self._boot_logger: logging.Logger = BootLogger().logger
        self._actors: OrderedDict[str, Actor] = OrderedDict()
        self._pipline_result: OrderedDict[str, dict] = OrderedDict()
        self._train_result: OrderedDict[str, TrainResult] = OrderedDict()
        self._PIPELINE_MAX = 1
        self._lock = Lock()

    def init(self) -> int:
        self._boot_logger.info("(" + self._worker + ") " + "init shared_state actor...")
        self._boot_logger.info("(" + self._worker + ") " + "set global logger...")
        self._logger = ray.get_actor(Actors.LOGGER)
        self._boot_logger.info("(" + self._worker + ") " + "set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._PIPELINE_MAX = int(config_parser.get("PIPELINE", "PIPELINE_MAX"))
        except configparser.Error as e:
            self._boot_logger.error("(" + self._worker + ") " + "an error occur when set config...: " + str(e))
            return -1
        self._boot_logger.info("(" + self._worker + ") " + "init shared_state actor complete...")
        return 0

    def set_actor(self, name: str, act: actor, state: int) -> None | int:
        self._lock.acquire()
        self._actors[name] = Actor(name=name, act=act, state=state)
        self._train_result[name] = TrainResult()
        self._lock.release()
        # split concurrency max, pipeline max
        # actors is only manage actor
        # result is only manage result and each max, pop last
        if len(self._actors) > self._PIPELINE_MAX:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="max piepline exceeded")
            return -1
        if len(self._pipline_result) >= self._PIPELINE_MAX:
            self._pipline_result.popitem(last=False)
            self._train_result.popitem(last=False)
        return 0

    def update_actor_state(self, name: str, state: int):
        if name in self._actors:
            act = self._actors[name]
            act.state = state

    def get_actor_state(self, name: str):
        if name in self._actors:
            act = self._actors[name]
            return act.state
        else:
            return None

    def is_actor_exist(self, name) -> bool:
        if name in self._actors:
            return True
        return False

    def delete_actor(self, name: str) -> None:
        if name in self._actors:
            del self._actors[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="actor not exist: " + name)

    def kill_actor(self, name: str) -> int:
        if name in self._actors:
            act = self._actors[name]
            ray.kill(act.act)
            self.delete_actor(name)
            return 0
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="actor not exist: " + name)
            return -1

    def set_pipeline_result(self, name: str, pipe_result: dict) -> None:
        self._pipline_result[name] = pipe_result

    def delete_pipeline_result(self, name: str) -> None:
        if name in self._pipline_result:
            del self._pipline_result[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="pipeline not exist: " + name)

    def get_pipeline_result(self, name: str) -> dict:
        if name in self._pipline_result:
            pipeline = self._pipline_result[name]
            return {"component_list": pipeline}
        else:
            return {}

    def set_train_progress(self, name: str, epoch: str, progress: str) -> None:
        self._train_result[name].set_train_progress(epoch=epoch, progress=progress)

    def set_test_progress(self, name: str, progress: str) -> None:
        self._train_result[name].set_test_progress(progress=progress)

    def set_train_result(self, name: str, train_result: dict) -> None:
        self._train_result[name].set_train_result(train_result)

    def set_test_result(self, name: str, test_result: dict) -> None:
        self._train_result[name].set_test_result(test_result)

    def get_train_result(self, name: str) -> dict:
        if name in self._train_result:
            train_result = self._train_result[name]
            return {"progress": train_result.get_train_progress(), "train_result": train_result.get_train_result(),
                    "test_result": train_result.get_test_result()}
        else:
            return {}

    def delete_train_result(self, name: str) -> None:
        if name in self._train_result:
            del self._train_result[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="pipeline not exist: " + name)


@dataclass
class Actor:
    name: str
    act: actor
    state: int
