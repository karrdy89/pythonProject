# *********************************************************************************************************************
# Program Name : shared_state
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import ast
import configparser
import logging
import requests
import json
from collections import OrderedDict
from threading import Lock
from datetime import datetime, timedelta

import ray
from ray import actor
from apscheduler.schedulers.background import BackgroundScheduler

from pipeline import TrainResult
from logger import BootLogger
from statics import Actors, TrainStateCode, ROOT_DIR
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
        self._error_message: OrderedDict[str, str] = OrderedDict()
        self._dataset_url: dict[str, str] = {}
        self._EXPIRE_TIME_DS_DOWNLOAD: int = 3600
        self._EXPIRE_TIME_TB: int = 3600
        self._tensorboard_tool = TensorBoardTool()
        self._session_id: dict[int, str] = {}
        self._SESSION_VALIDATION_URL = ''
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
            config_parser.read(ROOT_DIR + "/config/config.ini")
            self._PIPELINE_MAX = int(config_parser.get("PIPELINE", "PIPELINE_MAX"))
            self._DATASET_CONCURRENCY_MAX = int(config_parser.get("DATASET_MAKER", "MAX_CONCURRENCY"))
            self._URL_UPDATE_STATE_LRN = str(config_parser.get("MANAGE_SERVER", "URL_UPDATE_STATE_LRN"))
            self._SESSION_VALIDATION_URL = str(config_parser.get("MANAGE_SERVER", "SESSION_VALIDATION_URL"))
            self._EXPIRE_TIME_TB = int(config_parser.get("TENSOR_BOARD", "EXPIRE_TIME"))
        except configparser.Error as e:
            self._boot_logger.error("(" + self._worker + ") " + "an error occur when set config...: " + str(e))
            return -1
        self._scheduler.add_job(self._remove_garbage_tensorboard, "interval", minutes=10)
        self._boot_logger.info("(" + self._worker + ") " + "init shared_state actor complete...")
        return 0

    def set_actor(self, name: str, act: actor) -> None | int:
        if len(self._actors) + 1 > self._PIPELINE_MAX:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="max piepline exceeded")
            return -1
        else:
            self._actors[name] = act
            if name in self._train_result:
                del self._train_result[name]
            if name in self._pipline_result:
                del self._pipline_result[name]
            return 0

    def get_actor(self, name: str) -> actor:
        if name in self._actors:
            return self._actors[name]
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
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="delete actor: actor not exist: " + name)

    def kill_actor(self, name: str) -> int:
        if name in self._actors:
            act = self._actors[name]
            try:
                ray.kill(act)
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="kill actor: failed to kill: " + name + ':' + exc.__str__())
                return -1
            else:
                self.delete_actor(name)
                return 0
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="kill actor: actor not exist: " + name)
            return -2

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

    def set_train_status(self, name: str, user_id: str, status_code: int) -> None:
        if name not in self._train_result:
            if len(self._train_result) > self._PIPELINE_MAX:
                self._train_result.popitem(last=False)
            self._train_result[name] = TrainResult()
        self._train_result[name].set_train_status(status=status_code)
        if status_code == TrainStateCode.TRAINING_FAIL or status_code == TrainStateCode.TRAINING_DONE:
            self.send_state_code(name=name, user_id=user_id, state_code=status_code)

    def set_train_progress(self, name: str, epoch: str, progress: float, total_progress: float) -> None:
        if name not in self._train_result:
            if len(self._train_result) > self._PIPELINE_MAX:
                self._train_result.popitem(last=False)
            self._train_result[name] = TrainResult()
        self._train_result[name].set_train_progress(epoch=epoch, progress=progress,
                                                    total_progress=total_progress)

    def set_test_progress(self, name: str, progress: float) -> None:
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
            return {"train_progress": train_result.get_train_progress(),
                    "test_progress": train_result.get_test_progress(),
                    "train_result": train_result.get_train_result(),
                    "test_result": train_result.get_test_result()}
        else:
            return {}

    def delete_train_result(self, name: str) -> None:
        if name in self._train_result:
            del self._train_result[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="pipeline not exist: " + name)

    def set_make_dataset_result(self, name: str, user_id: str, state_code: int) -> None:
        if name not in self._make_dataset_result:
            if len(self._make_dataset_result) > self._DATASET_CONCURRENCY_MAX:
                self._make_dataset_result.popitem(last=False)
        self._make_dataset_result[name] = state_code
        if state_code == TrainStateCode.MAKING_DATASET_DONE or state_code == TrainStateCode.MAKING_DATASET_FAIL:
            self.send_state_code(name=name, user_id=user_id, state_code=state_code)

    def get_make_dataset_result(self, name: str) -> int:
        if name in self._make_dataset_result:
            return self._make_dataset_result[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="make dataset result not exist: " + name)

    def set_error_message(self, name: str, msg: str) -> None:
        if name not in self._error_message:
            if len(self._error_message) > self._DATASET_CONCURRENCY_MAX:
                self._error_message.popitem(last=False)
        self._error_message[name] = msg

    def get_error_message(self, name: str) -> str:
        if name in self._error_message:
            return self._error_message[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="task not exist: " + name)
            return ""

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
        run_date = datetime.now() + timedelta(seconds=self._EXPIRE_TIME_DS_DOWNLOAD)
        self._dataset_url[uid] = path
        self._scheduler.add_job(self.delete_dataset_url, "date", run_date=run_date, args=[uid])

    def get_dataset_path(self, uid) -> str | None:
        path = None
        if uid in self._dataset_url:
            path = self._dataset_url[uid]
            del self._dataset_url[uid]
        return path

    def delete_dataset_url(self, uid: str) -> None:
        if uid in self._dataset_url:
            del self._dataset_url[uid]

    def get_tensorboard_port(self, dir_path: str, session_id: str) -> int:
        port = self._tensorboard_tool.run(dir_path=dir_path)
        self._lock.acquire()
        self._session_id[port] = session_id
        self._lock.release()
        return port

    def get_session(self, port: str) -> tuple | None:
        return self._session_id.get(int(port)), self._SESSION_VALIDATION_URL

    def _expire_tensorboard(self, port: int, index: int) -> None:
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="_expire_tensorboard: activate")
        result = self._tensorboard_tool.expire_tensorboard(port=port, index=index)
        if result == 0:
            self._remove_session(port)
        return

    def _remove_garbage_tensorboard(self):
        self._logger.log.remote(level=logging.DEBUG, worker=self._worker,
                                msg="_remove_garbage_tensorboard: activate")
        idx = 0
        for port, session_id in self._session_id.items():
            data = {"SESSION_ID": session_id}
            headers = {'Content-Type': 'application/json; charset=utf-8'}
            try:
                res = requests.post(self._SESSION_VALIDATION_URL, data=json.dumps(data), headers=headers, timeout=10)
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="remove_garbage_tensorboard: " + exc.__str__())
                return
            else:
                if res.status_code == 200:
                    body = ast.literal_eval(res.content.decode('utf-8'))
                    is_valid = body.get("IS_VALID_SESSION_ID")
                    if is_valid == 'Y':
                        continue
                    else:
                        self._expire_tensorboard(port, idx)
            idx += 1

    def _remove_session(self, port: int) -> None:
        if port in self._session_id:
            del self._session_id[port]

    def send_state_code(self, name: str, user_id: str, state_code: int) -> None:
        sp_nm = name.split(':')
        mdl_id = sp_nm[0]
        sp_version = sp_nm[-1].split('.')
        mn_ver = sp_version[0]
        n_ver = sp_version[1]
        data = {"MDL_ID": mdl_id, "MN_VER": mn_ver, "N_VER": n_ver,
                "MDL_LRNG_ST_CD": str(state_code), "USR_ID": user_id}
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        try:
            res = requests.post(self._URL_UPDATE_STATE_LRN, data=json.dumps(data), headers=headers, timeout=10)
        except Exception as e:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="http request fail: update train state" + e.__str__())
        else:
            if res.status_code == 200:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="http request success: update train state")
            else:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="http request fail: update train state: "
                                            + str(res.status_code))

