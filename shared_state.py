import configparser
import logging
import requests
import traceback
import json
from collections import OrderedDict
from threading import Lock
from datetime import datetime, timedelta

import ray
from ray import actor
from apscheduler.schedulers.background import BackgroundScheduler

from pipeline import TrainResult
from logger import BootLogger
from statics import Actors, TrainStateCode, BuiltinModels
from tensorboard_service import TensorBoardTool


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
        self._actors: OrderedDict[str, actor] = OrderedDict()
        self._pipline_result: OrderedDict[str, dict] = OrderedDict()
        self._train_result: OrderedDict[str, TrainResult] = OrderedDict()
        self._make_dataset_result: OrderedDict[str, int] = OrderedDict()
        self._dataset_url: dict[str, str] = {}
        self._EXPIRE_TIME: int = 3600
        self._tensorboard_tool = TensorBoardTool()
        self._PIPELINE_MAX = 1
        self._DATASET_CONCURRENCY_MAX = 1
        self._URL_UPDATE_STATE_LRN = ''
        self._lock = Lock()
        self._scheduler = BackgroundScheduler()
        self._scheduler.start()

    def init(self) -> int:
        self._boot_logger.info("(" + self._worker + ") " + "init shared_state actor...")
        self._boot_logger.info("(" + self._worker + ") " + "set global logger...")
        self._logger = ray.get_actor(Actors.LOGGER)
        self._boot_logger.info("(" + self._worker + ") " + "set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._PIPELINE_MAX = int(config_parser.get("PIPELINE", "PIPELINE_MAX"))
            self._DATASET_CONCURRENCY_MAX = int(config_parser.get("DATASET_MAKER", "MAX_CONCURRENCY"))
            self._URL_UPDATE_STATE_LRN = str(config_parser.get("MANAGE_SERVER", "URL_UPDATE_STATE_LRN"))
        except configparser.Error as e:
            self._boot_logger.error("(" + self._worker + ") " + "an error occur when set config...: " + str(e))
            return -1
        self._boot_logger.info("(" + self._worker + ") " + "init shared_state actor complete...")
        return 0

    def set_actor(self, name: str, act: actor) -> None | int:
        if len(self._actors) + 1 > self._PIPELINE_MAX:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="max piepline exceeded")
            return -1
        else:
            self._lock.acquire()
            self._actors[name] = act
            self._lock.release()
            return 0

    def is_actor_exist(self, name) -> bool:
        if name in self._actors:
            return True
        return False

    def delete_actor(self, name: str) -> None:
        if name in self._actors:
            del self._actors[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="delete actor: actor not exist: " + name)

    def kill_actor(self, name: str) -> int:
        if name in self._actors:
            act = self._actors[name]
            ray.kill(act)
            self.delete_actor(name)
            return 0
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="kill actor: actor not exist: " + name)
            return -1

    def set_pipeline_result(self, name: str, pipe_result: dict) -> None:
        if name not in self._pipline_result:
            if len(self._pipline_result) > self._PIPELINE_MAX:
                self._pipline_result.popitem(last=False)
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

    def set_train_status(self, name: str, status_code: int) -> None:
        if name not in self._train_result:
            if len(self._train_result) > self._PIPELINE_MAX:
                self._train_result.popitem(last=False)
            self._train_result[name] = TrainResult()
        self._train_result[name].set_train_status(status=status_code)

    def set_train_progress(self, name: str, epoch: str, progress: str) -> None:
        if name not in self._train_result:
            if len(self._train_result) > self._PIPELINE_MAX:
                self._train_result.popitem(last=False)
            self._train_result[name] = TrainResult()
        self._train_result[name].set_train_progress(epoch=epoch, progress=progress)

    def set_test_progress(self, name: str, progress: str) -> None:
        if name not in self._train_result:
            if len(self._train_result) > self._PIPELINE_MAX:
                self._train_result.popitem(last=False)
            self._train_result[name] = TrainResult()
        self._train_result[name].set_test_progress(progress=progress)

    def set_train_result(self, name: str, train_result: dict) -> None:
        if name not in self._train_result:
            if len(self._train_result) > self._PIPELINE_MAX:
                self._train_result.popitem(last=False)
            self._train_result[name] = TrainResult()
        self._train_result[name].set_train_result(train_result)

    def set_test_result(self, name: str, test_result: dict) -> None:
        if name not in self._train_result:
            if len(self._train_result) > self._PIPELINE_MAX:
                self._train_result.popitem(last=False)
            self._train_result[name] = TrainResult()
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

    def set_make_dataset_result(self, name: str, state_code: int) -> None:
        if name not in self._make_dataset_result:
            if len(self._make_dataset_result) > self._DATASET_CONCURRENCY_MAX:
                self._make_dataset_result.popitem(last=False)
        self._make_dataset_result[name] = state_code
        if state_code is TrainStateCode.TRAINING_DONE or TrainStateCode.TRAINING_FAIL:
            sp_nm = name.split(':')
            mdl_id = sp_nm[0]
            sp_version = sp_nm[-1].split('.')
            mn_ver = sp_version[0]
            n_ver = sp_version[1]
            try:
                data = {"MDL_ID": mdl_id, "MN_VER": mn_ver, "N_VER": n_ver, "MDL_LRNG_ST_CD": str(state_code)}
                headers = {'Content-Type': 'application/json; charset=utf-8'}
                res = requests.post(self._URL_UPDATE_STATE_LRN, data=json.dumps(data), headers=headers)
            except Exception as e:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="http request fail: update make dataset state" + e.__str__())
            else:
                if res.status_code == 200:
                    self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                            msg="http request success: update make dataset state")
                else:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                            msg="http request fail: update make dataset state code: "
                                                + str(res.status_code))

    def get_make_dataset_result(self, name: str) -> int:
        if name in self._make_dataset_result:
            return self._make_dataset_result[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="make dataset result not exist: " + name)

    def delete_make_dataset_result(self, name: str) -> None:
        if name in self._make_dataset_result:
            del self._make_dataset_result[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="make dataset result not exist: " + name)

    def get_status_code(self, name: str) -> dict:
        result = {"DATASET": "", "TRAIN": ""}
        if name in self._train_result:
            result["TRAIN"] = self._train_result[name].get_train_status()
        if name in self._make_dataset_result:
            result["DATASET"] = self._make_dataset_result[name]
        return result

    def set_dataset_url(self, uid: str, path: str) -> None:
        run_date = datetime.now() + timedelta(seconds=self._EXPIRE_TIME)
        self._lock.acquire()
        self._dataset_url[uid] = path
        self._lock.release()
        self._scheduler.add_job(self.delete_dataset_url, "date", run_date=run_date, args=[uid])

    def get_dataset_path(self, uid) -> str | None:
        self._lock.acquire()
        path = None
        if uid in self._dataset_url:
            path = self._dataset_url[uid]
            del self._dataset_url[uid]
        self._lock.release()
        return path

    def delete_dataset_url(self, uid: str) -> None:
        self._lock.acquire()
        if uid in self._dataset_url:
            del self._dataset_url[uid]
        self._lock.release()
        print(self._dataset_url)

    def get_tensorboard_port(self, dir_path: str) -> int:
        self._lock.acquire()
        port = self._tensorboard_tool.run(dir_path=dir_path)
        self._lock.release()
        return port
