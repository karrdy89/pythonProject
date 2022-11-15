# *********************************************************************************************************************
# Program Name : onnx_serving_manager
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import configparser
import uuid
import logging
import logger
from dataclasses import dataclass, field
from threading import Lock
from itertools import cycle

import ray
from apscheduler.schedulers.background import BackgroundScheduler

from utils.common import version_encode
from logger import BootLogger
from statics import Actors, BuiltinModels, ModelType
from db import DBUtil
from onnx_serving import OnnxServing


@ray.remote
class OnnxServingManager:
    """
    A ray actor class to deploy and manage onnx model

    Attributes
    ----------
    _worker : str
        The class name of instance.
    _logger : actor
        A Logger class for logging
    _boot_logger : logger
        The pre-defined Logger class for logging init process.
    _db : DBUtil
        The DBUtil for read deploy state table
    _lock : Lock
        Thread lock for synchronization.
    _deploy_requests : list[tuple[str, str]]
        A list of deploy requests. if deploy done, the request will remove from this list.
        (Ex. [(model_1, version), (model_2, version), ...])
    _deploy_states : dict[str, ModelDeployState]
        A dictionary of current deploy state.
        (Ex. {model1_version: ModelDeployState}, {model2_version: ModelDeployState})
    _gc_list : list[tuple[int, str]]
        A list of container to delete GC will remove container with this list.
        (Ex. [(ManageType, model1_version), (ManageType, model2_version), ...])
    _manager_handle : AsyncIOScheduler
        An AsyncIOScheduler instance for managing container
    _GC_CHECK_INTERVAL : int
        Configuration of interval of managing container
    """
    def __init__(self):
        self._worker: str = type(self).__name__
        self._logger: ray.actor = None
        self._shared_state: ray.actor = None
        self._boot_logger: logger = BootLogger().logger
        self._db = None
        self._lock = Lock()
        self._deploy_requests: list[tuple[str, str]] = []
        self._deploy_states: dict[str, ModelDeployState] = {}
        self._gc_list: list[tuple[int, str]] = []
        self._manager_handle: BackgroundScheduler | None = None
        self._MAX_DEPLOY: int = 20
        self._GC_CHECK_INTERVAL: int = 10

    def init(self) -> int:
        self._boot_logger.info("(" + self._worker + ") " + "init onnx_serving actor...")
        self._boot_logger.info("(" + self._worker + ") " + "set global logger...")
        self._logger = ray.get_actor(Actors.LOGGER)
        self._shared_state = ray.get_actor(Actors.GLOBAL_STATE)
        self._boot_logger.info("(" + self._worker + ") " + "set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._MAX_DEPLOY = int(config_parser.get("DEPLOY", "MAX_DEPLOY"))
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
                    result = self.deploy_actor(model_id, version, deploy_num, model_deploy_state)
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
            self._lock.release()
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
                diff = deploy_num - len(model_deploy_state.actors)
                if diff > 0:
                    try:
                        result = self.add_actor(model_id, version, diff)
                    except Exception as exc:
                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                msg="deploy error : " + model_id
                                                    + ":" + version + ":" + exc.__str__())
                elif diff < 0:
                    try:
                        result = self.remove_actor(model_id, version, abs(diff))
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
                    result = self.deploy_actor(model_id=model_id, version=version,
                                               deploy_num=deploy_num, model_deploy_state=model_deploy_state)
                except Exception as exc:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                            msg="deploy error : " + model_id
                                                + ":" + version + ":" + exc.__str__())
            self._deploy_requests.remove((model_id, version))
            return result

    def deploy_actor(self, model_id: str, version: str, deploy_num: int, model_deploy_state) -> dict:
        set_deploy_num_result = ray.get(self._shared_state.set_deploy_num.remote(diff=deploy_num))
        if set_deploy_num_result == -1:
            current_deploy_num = ray.get(self._shared_state.get_deploy_num.remote())
            result = {"CODE": "FAIL", "ERROR_MSG": "max serving actor exceeded",
                      "MSG": {"current serving actor": current_deploy_num,
                              "max serving actor": self._MAX_DEPLOY}}
            return result
        model_key = model_id + "_" + version
        actors = []
        actor_names = []
        for _ in range(deploy_num):
            name = model_id + "-" + uuid.uuid4().hex
            onnx_server = OnnxServing.options(name=name).remote()
            actor_names.append(name)
            actors.append(onnx_server)
        init_result = ray.get([onx_serv.init.remote(model_id=model_id, version=version) for onx_serv in actors])
        deploy_count = 0
        error_msg = ""
        for i, res in enumerate(init_result):
            code = res[0]
            msg = res[1]
            if code == -1:
                error_msg = msg
            else:
                serving_actor = ServingActor(name=actor_names[i], actor=actors[i], state=StateCode.AVAILABLE)
                model_deploy_state.actors[actor_names[i]] = serving_actor
                deploy_count += 1

        if deploy_count != deploy_num:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="can create serving actor : " + error_msg + ":" + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "can create serving actor: " + error_msg, "MSG": ""}
            for actor in actors:
                ray.kill(actor)
            return result
        else:
            self._deploy_states[model_key] = model_deploy_state
            self._set_cycle(model_id=model_id, version=version)
            result = {"CODE": "SUCCESS", "ERROR_MSG": "",
                      "MSG": "deploy finished. " + str(deploy_count) + "/" + str(deploy_num) + " deployed"}
            return result

    def end_deploy(self, model_id: str, version: str) -> dict:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get end deploy request")
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        result = None
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="model not deployed : "
                                                                                 + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "the model is not deployed", "MSG": ""}
        else:
            if model_deploy_state.state == StateCode.SHUTDOWN:
                result = {"CODE": "FAIL", "ERROR_MSG": "shutdown has already been scheduled", "MSG": ""}
            elif model_deploy_state.state == StateCode.AVAILABLE:
                self._lock.acquire()
                model_deploy_state.state = StateCode.SHUTDOWN
                self._lock.release()
                self._gc_list.append((ManageType.MODEL, model_key))
                ray.get(self._shared_state.set_deploy_num.remote(diff=-len(model_deploy_state.actors)))
                self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="end deploy : "
                                                                                     + model_id + ":" + version)
                result = {"CODE": "SUCCESS", "ERROR_MSG": "", "MSG": "end deploy accepted"}
        return result

    def get_deploy_state(self) -> dict:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get deploy state")
        current_deploy_num = ray.get(self._shared_state.get_deploy_num.remote())
        total_deploy_state = []
        for model_key in self._deploy_states:
            actors = []
            sep_key = model_key.split("_")
            model_id = sep_key[0]
            version = sep_key[1]
            model_deploy_state = self._deploy_states[model_key]
            for k, v in model_deploy_state.actors.items():
                actors.append((v.name, v.state))
            deploy_state = {"model_id": model_id, "version": version, "actors": actors, "deploy_num": len(actors)}
            total_deploy_state.append(deploy_state)
        return {"CODE": "SUCCESS", "ERROR_MSG": "", "DEPLOY_STATE": total_deploy_state,
                "CURRENT_DEPLOY_NUM": current_deploy_num}

    def add_actor(self, model_id: str, version: str, deploy_num: int) -> dict:
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="the model is not deployed : "
                                                                                 + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "the model is not deployed", "MSG": ""}
            return result
        result = self.deploy_actor(model_id, version, deploy_num, model_deploy_state)
        return result

    def remove_actor(self, model_id: str, version: str, deploy_num: int) -> dict:
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="the model is not deployed : "
                                                                                 + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "the model is not deployed", "MSG": ""}
            return result

        if model_deploy_state.state == StateCode.SHUTDOWN:
            return {"CODE": "FAIL", "ERROR_MSG": "shutdown has already been scheduled on this model", "MSG": ""}

        actors = model_deploy_state.actors
        available_actors = []
        for k, v in actors.items():
            if v.state == StateCode.AVAILABLE:
                available_actors.append(v.actor)

        if deploy_num >= len(available_actors):
            result = self.end_deploy(model_id, version)
            return result

        gc_list = []
        for i, key in enumerate(actors):
            if i < deploy_num:
                container = actors[key]
                if container.state == StateCode.AVAILABLE:
                    container.state = StateCode.SHUTDOWN
                    gc_list.append((ManageType.ACTOR, key))
            else:
                break

        result = self._set_cycle(model_id, version)
        if result == 0:
            self._gc_list = self._gc_list + gc_list
            self._shared_state.set_deploy_num.remote(-len(gc_list))
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="shutdown pending on " + str(deploy_num) + " containers : " + model_id + ":"
                                        + version)
            result = {"CODE": "SUCCESS", "ERROR_MSG": "",
                      "MSG": "shutdown pending on " + str(deploy_num) + " containers"}
            return result
        else:
            result = {"CODE": "FAIL", "ERROR_MSG": "failed to remove container", "MSG": ""}
            return result

    def _set_cycle(self, model_id: str, version: str) -> int:
        cycle_list = []
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        actors = model_deploy_state.actors
        self._lock.acquire()
        for key in actors:
            actor = actors[key]
            cycle_list.append([actor.actor, actor.state, actor.name])
        model_deploy_state.cycle_iterator = cycle(cycle_list)
        self._lock.release()
        return 0

    def fail_back(self, model_id: str, version: str, actor_name: str) -> None:
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="trying to fail back container : " + model_id + ":" + version)
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if model_deploy_state is not None:
            if model_deploy_state.state != StateCode.SHUTDOWN and model_deploy_state.state != StateCode.UNAVAILABLE:
                actor = model_deploy_state.actors.get(actor_name)
                actor.state = StateCode.SHUTDOWN
                deploy_total = len(model_deploy_state.actors)
                unavailable_container = 0
                for container_key, actor in model_deploy_state.actors.items():
                    if actor.state == StateCode.SHUTDOWN:
                        unavailable_container += 1
                if deploy_total == unavailable_container:
                    model_deploy_state.state = StateCode.UNAVAILABLE
                result = self._set_cycle(model_id, version)
                if result == 0:
                    self._gc_list.append((ManageType.ACTOR, actor_name))
                result_add_container = self.add_actor(model_id, version, 1)
                if result_add_container.get("CODE") == "SUCCESS":
                    model_deploy_state.state = StateCode.AVAILABLE

    def gc_actor(self) -> None:
        self._logger.log.remote(level=logging.DEBUG, worker=self._worker, msg="GC activate")
        for type_key in self._gc_list.copy():
            manage_type = type_key[0]
            key = type_key[1]
            if manage_type == ManageType.MODEL:
                model_deploy_state = self._deploy_states.get(key)
                actors = model_deploy_state.actors
                model_ref_count = 0
                for actor_name, serving_actor in actors.items():
                    model_ref_count += serving_actor.ref_count
                if model_ref_count == 0:
                    remove_count = 0
                    for actor_name, serving_actor in actors.items():
                        actor = serving_actor.actor
                        try:
                            ray.kill(actor)
                        except Exception as exc:
                            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                    msg="an error occur when remove actor : " + exc.__str__())
                            continue
                        remove_count += 1
                    del self._deploy_states[key]
                    self._gc_list.remove(type_key)
                    self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                            msg="model deploy ended: " + key)
            elif manage_type == ManageType.ACTOR:
                for mds_key in self._deploy_states:
                    model_deploy_state = self._deploy_states[mds_key]
                    actors = model_deploy_state.actors
                    if key in actors:
                        sep = mds_key.split("_")
                        model_id = sep[0]
                        version = sep[1]
                        if len(actors) > 1:
                            actor = actors.get(key)
                            if actor.ref_count <= 0:
                                try:
                                    ray.kill(actor.actor)
                                except Exception as exc:
                                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                            msg="an error occur when remove actor : " + exc.__str__())
                                    continue
                                del actors[key]
                                self._set_cycle(model_id, version)
                                self._gc_list.remove(type_key)
                                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                                        msg="actor deleted : " + key)
                        else:
                            self.end_deploy(model_id, version)
                            self._gc_list.remove(type_key)
                    else:
                        continue

    def predict(self, model_id: str, version: str, data: list) -> dict:
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        result = {"CODE": "FAIL", "ERROR_MSG": "N/A", "EVNT_ID": [], "PRBT": []}
        if (model_deploy_state is None) or (model_deploy_state.cycle_iterator is None):
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="model not deployed : "
                                                                                 + model_id + ":" + version)
            result["CODE"] = "FAIL"
            result["ERROR_MSG"] = "the model is not deployed"
            return result

        if model_deploy_state.state == StateCode.SHUTDOWN:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="model not deployed : "
                                                                                 + model_id + ":" + version)
            result["CODE"] = "FAIL"
            result["ERROR_MSG"] = "the model is not deployed"
            return result
        elif model_deploy_state.state == StateCode.UNAVAILABLE:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="all containers are currently un available : " + model_id + ":" + version)
            result["CODE"] = "FAIL"
            result["ERROR_MSG"] = "all actor are currently un available and attempting to fail back"
            return result

        deploy_info = next(model_deploy_state.cycle_iterator)
        actor = deploy_info[0]
        state = deploy_info[1]
        actor_name = deploy_info[2]
        serving_actor = model_deploy_state.actors[actor_name]
        if state == StateCode.AVAILABLE:
            serving_actor.ref_count += 1
            try:
                predict_result = ray.get(actor.predict.remote(data=data))
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="can't get inference from actor : " + exc.__str__() + model_id + ":" + version)
                self.fail_back(model_id, version, actor_name)
                serving_actor.ref_count -= 1
                result = self.predict(model_id, version, data)
            else:
                serving_actor.ref_count -= 1
                result = predict_result
            return result
        else:
            result = self.predict(model_id, version, data)
            return result


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
    UNAVAILABLE = -1
    SHUTDOWN = 4


class ManageType:
    MODEL = 0
    CONTAINER = 1
    ACTOR = 2
