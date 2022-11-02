import configparser
import os
import shutil
import uuid
import functools
import logging
import logger
from dataclasses import dataclass, field
from threading import Lock
from shutil import copytree, rmtree
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Optional

import ray
from apscheduler.schedulers.background import BackgroundScheduler

from utils.common import version_encode, version_decode
from logger import BootLogger
from statics import Actors, BuiltinModels, ModelType
from db import DBUtil


@ray.remote
class OnnxServingManager:
    def __init__(self):
        self._worker: str = type(self).__name__
        self._logger: ray.actor = None
        self._boot_logger: logger = BootLogger().logger
        self._db = None
        self._lock = Lock()
        self._deploy_requests: list[tuple[str, str]] = []
        self._deploy_states: dict[str, ModelDeployState] = {}
        self._gc_list: list[tuple[int, str]] = []
        self._current_deploy_num: int = 0
        self._manager_handle: BackgroundScheduler | None = None
        self._SERVING_ACTOR_MAX: int = 15
        self._GC_CHECK_INTERVAL: int = 10

    def init(self) -> int:
        self._boot_logger.info("(" + self._worker + ") " + "init onnx_serving actor...")
        self._boot_logger.info("(" + self._worker + ") " + "set global logger...")
        self._logger = ray.get_actor(Actors.LOGGER)
        self._boot_logger.info("(" + self._worker + ") " + "set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._SERVING_ACTOR_MAX = int(config_parser.get("DEPLOY", "SERVING_ACTOR_MAX"))
            self._GC_CHECK_INTERVAL = int(config_parser.get("DEPLOY", "GC_CHECK_INTERVAL"))
        except configparser.Error as e:
            self._boot_logger.error("(" + self._worker + ") " + "an error occur when set config...: " + str(e))
            return -1

        self._boot_logger.info("(" + self._worker + ") " + "set garbage container collector...")
        self._manager_handle = BackgroundScheduler()
        self._manager_handle.add_job(self.gc_actor, "interval", seconds=self._GC_CHECK_INTERVAL, id="gc_actor")
        self._manager_handle.start()

        self._boot_logger.info("(" + self._worker + ") " + "deploying existing models...")

        try:
            self._db = DBUtil()
        except Exception as exc:
            self._boot_logger.error(
                "(" + self._worker + ") " + "can't initiate DBUtil:" + exc.__str__())

        try:
            stored_deploy_states = self._db.select(name="select_deploy_state")
        except Exception as exc:
            self._boot_logger.error(
                "(" + self._worker + ") " + "can't read deploy state from db:" + exc.__str__())
            # build
            # return -1
        else:
            for stored_deploy_state in stored_deploy_states:
                model_id = stored_deploy_state[0]
                model_type = getattr(BuiltinModels, model_id)
                model_type = model_type.model_type
                if model_type == ModelType.ONNX:
                    mn_ver = str(stored_deploy_state[1])
                    n_ver = str(stored_deploy_state[2])
                    deploy_num = stored_deploy_state[3]
                    version = mn_ver + "." + n_ver
                    encoded_version = version_encode(version)
                    model_deploy_state = ModelDeployState(model=(model_id, encoded_version),
                                                          state=StateCode.AVAILABLE)
                    result = await self.deploy_actor(model_id, version, deploy_num, model_deploy_state)
                    if result.get("CODE") == "FAIL":
                        self._boot_logger.error(
                            "(" + self._worker + ") " + "can't make actor from stored deploy state")
                        return -1
        self._boot_logger.info("(" + self._worker + ") " + "init onnx_serving actor complete...")
        return 0

    def deploy(self, model_id: str, version: str, deploy_num: int) -> dict:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="deploy start : " + model_id
                                                                             + ":" + version)
        self._lock.acquire()
        if (model_id, version) in self._deploy_requests:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="same deploy request is in progress : " + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "same deploy request is in progress", "MSG": ""}
            return result
        else:
            self._deploy_requests.append((model_id, version))
        self._lock.release()

        encoded_version = version_encode(version)
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        result = None
        if model_deploy_state is not None:
            if model_deploy_state.state == StateCode.AVAILABLE:
                diff = deploy_num - self._current_container_num
                if diff > 0:
                    try:
                        result = await self.add_actor(model_id, version, diff)
                    except Exception as exc:
                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                msg="deploy error : " + model_id
                                                    + ":" + version + ":" + exc.__str__())
                elif diff < 0:
                    try:
                        result = await self.remove_actor(model_id, version, abs(diff))
                    except Exception as exc:
                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                msg="deploy error : " + model_id
                                                    + ":" + version + ":" + exc.__str__())
                else:
                    result = {"CODE": "SUCCESS", "ERROR_MSG": "", "MSG": "nothing to change"}
            elif model_deploy_state.state == StateCode.SHUTDOWN:
                result = {"CODE": "FAIL", "ERROR_MSG": "shutdown has been scheduled on this model", "MSG": ""}
            self._deploy_requests.remove((model_id, version))
            return result
        else:
            if deploy_num <= 0:
                result = {"CODE": "SUCCESS", "ERROR_MSG": "", "MSG": "nothing to change"}
            else:
                model_deploy_state = ModelDeployState(model=(model_id, encoded_version),
                                                      state=StateCode.AVAILABLE)
                try:
                    result = self.deploy_actor(model_id, version, deploy_num, model_deploy_state)
                except Exception as exc:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                            msg="deploy error : " + model_id
                                                + ":" + version + ":" + exc.__str__())
            self._deploy_requests.remove((model_id, version))
            return result

    def deploy_actor(self, model_id: str, version: str, deploy_num: int, model_deploy_state) -> dict:
        pass

# manage actor like tf container
# method : deploy
# method : end_deploy
# method : get deploy state
# method : add actor
# method : remove actor
# method : set cycle
# method : deploy actor
# method : fail back
# method : garbage collect
# method : predict
# merge with tf serving? issue is max container num, init process, manage point
# only advantage is performance



@dataclass
class ServingActor:
    name: str
    actor: ray.actor
    state: int
    ref_count: int = 0


@dataclass
class ModelDeployState:
    model: tuple[str, int]
    state: int
    cycle_iterator = None
    actors: dict = field(default_factory=dict)


class StateCode:
    ALREADY_EXIST = 1
    IN_PROGRESS = 3
    AVAILABLE = 0
    UN_AVAILABLE = -1
    SHUTDOWN = 4


class ManageType:
    MODEL = 0
    CONTAINER = 1
