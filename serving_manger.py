# *********************************************************************************************************************
# Program Name : tf_serving_manager
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import configparser
import asyncio
import os
import uuid
import functools
import logging
import logger
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Optional

import docker
import ray
import grpc
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.config import model_server_config_pb2
from docker.errors import ContainerError, ImageNotFound, APIError, DockerException
from docker.models.containers import Container
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from utils.http_util import Http
from utils.common import version_encode, version_decode
from logger import BootLogger
from statics import Actors, ModelType, ROOT_DIR
from db import DBUtil
from onnx_serving import OnnxServing


@ray.remote
class ServingManager:
    """
    A ray actor class to serve and inference tensorflow model

    Attributes
    ----------
    _worker : str
        The class name of instance.
    _logger : actor
        A Logger class for logging
    _boot_logger : logger
        The pre-defined Logger class for logging init process.
    _lock : Lock
        Async lock for synchronization.
    _executor : ThreadPoolExecutor
        A ThreadPoolExecutor for parallel processing.
    _deploy_requests : list[tuple[str, str]]
        A list of deploy requests. if deploy done, the request will remove from this list.
        (Ex. [(model_1, version), (model_2, version), ...])
    _deploy_states : dict[str, ModelDeployState]
        A dictionary of current deploy state.
        (Ex. {model1_version: ModelDeployState}, {model2_version: ModelDeployState})
    _gc_list : list[tuple[int, str]]
        A list of container to delete GC will remove container with this list.
        (Ex. [(ManageType, model1_version), (ManageType, model2_version), ...])
    _client : DockerClient
        A docker client instance.
    _http_port : list[int]
        A list of available http port bind to container.
    _http_port_use : list[int]
        A list of http port in use by container.
    _grpc_port : list[int]
        A list of available grpc port bind to container.
    _grpc_port_use : list[int]
        A list of grpc port in use by container.
    _manager_handle : AsyncIOScheduler
        An AsyncIOScheduler instance for managing container
    _HTTP_PORT_START : int
        Configuration of http port range
    _GRPC_PORT_START : int
        Configuration of grpc port range
    _GC_CHECK_INTERVAL : int
        Configuration of interval of managing container
    _DEPLOY_PATH : str
        Configuration of location that model has to locate for deploying
    _CONTAINER_IMAGE : str
        Configuration of tensorflow serving image
    _CONTAINER_SERVER_IP : str
        Configuration of docker server ip

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    init() -> int
        Set attributes.
    deploy(model_id: str, version: str, container_num: int) -> json
        Carry out deploy request
    get_deploy_state() -> json:
        Return _deploy_states.
    add_container(model_id: str, version: str, container_num: int) -> json:
        Add containers for the deployed model
    remove_container(model_id: str, version: str, container_num: int) -> json:
        Remove containers for the deployed model
    end_deploy(model_id: str, version: str) -> json:
        End model deployment
    predict(model_id: str, version: str, data: dict) -> json:
        Get inference from container with given data. and return it.
    deploy_containers(model_id: str, version: str, container_num: int, model_deploy_state):
        Request to create container in parallel with run_container.
    run_container(model_id: str, container_name: str, http_port: int, grpc_port: int, deploy_path: str)
                    -> Container | None:
        Request to create container with docker server.
    fail_back(model_id: str, version: str, container_name: str) -> None:
        Shutting down and restarting unconnected container.
    get_port_http() -> int:
        Get port number from available port list.
    get_port_grpc() -> int:
        Get port number from available port list.
    release_port_http(port: int) -> None:
        Release port number from list of port in use.
    release_port_grpc(port: int) -> None:
        Release port number from list of port in use.
    reset_version_config(host: str, name: str, base_path: str, model_platform: str, model_version_policy: list):
        Changing the settings of a running TensorFlow Serving Container via grpc.
    add_version(model_id: str, version: str):
        Changing model version of a running TensorFlow Serving Container.
    _set_cycle(model_id: str, version: str) -> int:
        Set cycle instance for round-robin.
    gc_container() -> None:
        Remove containers that need to be cleared.
    get_container_list():
        Get a list of currently running containers from the docker client.
    get_container_names():
        Get a name of list that currently running containers from the docker client.
    """

    def __init__(self):
        self._worker: str = type(self).__name__
        self._logger: ray.actor = None
        self._boot_logger: logger = BootLogger().logger
        self._db = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor()
        self._deploy_requests: list[tuple[str, str]] = []
        self._deploy_states: dict[str, ModelDeployState] = {}
        self._gc_list: list[tuple[int, str]] = []
        self._client: docker.DockerClient | None = None
        self._http_port: list[int] = []
        self._http_port_use: list[int] = []
        self._grpc_port: list[int] = []
        self._grpc_port_use: list[int] = []
        self._manager_handle: AsyncIOScheduler | None = None
        self._HTTP_PORT_START: int = 8500
        self._GRPC_PORT_START: int = 8000
        self._GC_CHECK_INTERVAL: int = 10
        self._DEPLOY_PATH: str = ''
        self._MAX_DEPLOY: int = 20
        self._CONTAINER_IMAGE: str = "tensorflow/serving:2.6.5"
        self._CONTAINER_SERVER_IP: str = ""
        self._RETRY_COUNT: int = 20
        self._RETRY_WAIT_TIME: float = 0.1
        self._deploy_num = 0

    async def init(self) -> int:
        self._boot_logger.info("(" + self._worker + ") " + "init tensorflow_serving manager actor...")
        self._boot_logger.info("(" + self._worker + ") " + "set global logger...")
        self._logger = ray.get_actor(Actors.LOGGER)
        self._boot_logger.info("(" + self._worker + ") " + "set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._HTTP_PORT_START = int(config_parser.get("DEPLOY", "HTTP_PORT_START"))
            self._GRPC_PORT_START = int(config_parser.get("DEPLOY", "GRPC_PORT_START"))
            self._GC_CHECK_INTERVAL = int(config_parser.get("DEPLOY", "GC_CHECK_INTERVAL"))
            self._DEPLOY_PATH = str(config_parser.get("DEPLOY", "DEPLOY_PATH"))
            self._MAX_DEPLOY = int(config_parser.get("DEPLOY", "MAX_DEPLOY"))
            self._CONTAINER_IMAGE = str(config_parser.get("DEPLOY", "CONTAINER_IMAGE"))
            self._CONTAINER_SERVER_IP = str(config_parser.get("DEPLOY", "CONTAINER_SERVER_IP"))
            self._RETRY_COUNT: int = int(config_parser.get("DEPLOY", "RETRY_COUNT"))
            self._RETRY_WAIT_TIME: float = float(config_parser.get("DEPLOY", "RETRY_WAIT_TIME"))
        except configparser.Error as e:
            self._boot_logger.error("(" + self._worker + ") " + "an error occur when set config...: " + str(e))
            return -1

        self._boot_logger.info("(" + self._worker + ") " + "set container port range...")
        for i in range(self._MAX_DEPLOY * 2):
            self._grpc_port.append(self._GRPC_PORT_START + i)
            self._http_port.append(self._HTTP_PORT_START + i)

        self._boot_logger.info("(" + self._worker + ") " + "set docker client...")
        try:
            self._client = docker.from_env()
        except DockerException as e:
            self._boot_logger.error("(" + self._worker + ") " + "can't make connection to docker client:" + e.__str__())
            return -1
        self._boot_logger.info("(" + self._worker + ") " + "remove all serving containers...")
        self.remove_all_container()

        self._boot_logger.info("(" + self._worker + ") " + "set garbage container collector...")
        self._manager_handle = AsyncIOScheduler()
        self._manager_handle.add_job(self.gc_container, "interval", seconds=self._GC_CHECK_INTERVAL, id="gc_container")
        self._manager_handle.start()

        self._boot_logger.info("(" + self._worker + ") " + "deploying models from db...")
        self._db = DBUtil()
        try:
            stored_deploy_states = self._db.select(name="select_deploy_state")
        except Exception as exc:
            self._boot_logger.error(
                "(" + self._worker + ") " + "can't read deploy state from db:" + exc.__str__())
            # return -1
        else:
            for stored_deploy_state in stored_deploy_states:
                model_id = stored_deploy_state[0]
                # model_type = getattr(BuiltinModels, model_id)
                # model_type = model_type.model_type
                model_type = stored_deploy_state[5]  # depends on query
                mn_ver = str(stored_deploy_state[1])
                n_ver = str(stored_deploy_state[2])
                deploy_num = stored_deploy_state[3]
                version = mn_ver + "." + n_ver
                encoded_version = version_encode(version)
                model_deploy_state = ModelDeployState(model=(model_id, encoded_version),
                                                      state=StateCode.AVAILABLE, model_type=model_type)
                result = await self.deploy_server(model_id, version, deploy_num, model_deploy_state, model_type)
                if result.get("CODE") == "FAIL":
                    self._boot_logger.error(
                        "(" + self._worker + ") " + "can't make actor from stored deploy state")
                    return -1

        self._boot_logger.info("(" + self._worker + ") " + "init tensorflow_serving manager actor complete...")
        return 0

    async def deploy(self, model_id: str, version: str, deploy_num: int, model_type: int) -> dict:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="deploy start : " + model_id
                                                                             + ":" + version)
        async with self._lock:
            if (model_id, version) in self._deploy_requests:
                self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                        msg="same deploy request is in progress : " + model_id + ":" + version)
                result = {"CODE": "FAIL", "ERROR_MSG": "same deploy request is in progress", "MSG": ""}
                return result
            else:
                self._deploy_requests.append((model_id, version))
        encoded_version = version_encode(version)
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        result = None
        if model_deploy_state is not None:
            if model_deploy_state.state == StateCode.AVAILABLE:
                deploy_num = abs(deploy_num)
                count = 0
                for k, v in model_deploy_state.servers.items():
                    if v.state == StateCode.AVAILABLE:
                        count += 1
                diff = deploy_num - count
                if diff > 0:
                    try:
                        result = await self.add_server(model_id, version, diff)
                    except Exception as exc:
                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                msg="deploy error : " + model_id
                                                    + ":" + version + ":" + exc.__str__())
                elif diff < 0:
                    try:
                        result = await self.remove_server(model_id, version, abs(diff))
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
                                                      state=StateCode.AVAILABLE, model_type=model_type)
                try:
                    result = await self.deploy_server(model_id, version, deploy_num, model_deploy_state, model_type)
                    print(result)
                except Exception as exc:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                            msg="deploy error : " + model_id
                                                + ":" + version + ":" + exc.__str__())
            self._deploy_requests.remove((model_id, version))
            return result

    async def get_deploy_state(self) -> dict:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get deploy state")
        total_deploy_state = []
        for model_key in self._deploy_states:
            servers = []
            sep_key = model_key.split("_")
            model_id = sep_key[0]
            version = sep_key[1]
            model_deploy_state = self._deploy_states[model_key]
            for k, v in model_deploy_state.servers.items():
                servers.append((v.name, v.state))
            deploy_state = {"model_id": model_id, "version": version, "inference_server": servers,
                            "deploy_num": len(servers)}
            total_deploy_state.append(deploy_state)
        return {"CODE": "SUCCESS", "ERROR_MSG": "", "DEPLOY_STATE": total_deploy_state,
                "CURRENT_DEPLOY_NUM": self._deploy_num, "MAX_DEPLOY": self._MAX_DEPLOY}

    async def add_server(self, model_id: str, version: str, deploy_num: int) -> dict:
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="the model is not deployed : "
                                                                                 + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "the model is not deployed", "MSG": ""}
            return result
        result = await self.deploy_server(model_id, version, deploy_num, model_deploy_state,
                                          model_deploy_state.model_type)
        return result

    def remove_all_container(self) -> None:
        containers = self._client.containers.list(filters={"ancestor": self._CONTAINER_IMAGE})
        for container in containers:
            container.remove(force=True)

    async def remove_server(self, model_id: str, version: str, num: int) -> dict:
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="the model is not deployed : "
                                                                                 + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "the model is not deployed", "MSG": ""}
            return result

        if model_deploy_state.state == StateCode.SHUTDOWN:
            return {"CODE": "FAIL", "ERROR_MSG": "shutdown has already been scheduled on this model", "MSG": ""}

        servers = model_deploy_state.servers
        available_servers = []
        for k, v in servers.items():
            if v.state == StateCode.AVAILABLE:
                available_servers.append(v.inference_server)

        if num >= len(available_servers):
            result = await self.end_deploy(model_id, version)
            return result

        gc_list = []
        for i, key in enumerate(servers):
            if i < num:
                server = servers[key]
                if server.state == StateCode.AVAILABLE:
                    server.state = StateCode.SHUTDOWN
                    gc_list.append((ManageType.SERVER, key))
            else:
                break

        result = await self._set_cycle(model_id, version)
        if result == 0:
            self._gc_list = self._gc_list + gc_list
            async with self._lock:
                self._deploy_num -= len(gc_list)
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="shutdown pending on " + str(num) + " containers : " + model_id + ":"
                                        + version)
            result = {"CODE": "SUCCESS", "ERROR_MSG": "",
                      "MSG": "shutdown pending on " + str(num) + " containers"}
            return result
        else:
            result = {"CODE": "FAIL", "ERROR_MSG": "failed to remove container", "MSG": ""}
            return result

    async def end_deploy(self, model_id: str, version: str) -> dict:
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        result = None
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="model not deployed : "
                                                                                 + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "the model is not deployed", "MSG": ""}
        else:
            if model_deploy_state.state == StateCode.SHUTDOWN:
                result = {"CODE": "FAIL", "ERROR_MSG": "shutdown has already been scheduled", "MSG": ""}
            elif model_deploy_state.state == StateCode.AVAILABLE:
                model_deploy_state.state = StateCode.SHUTDOWN
                async with self._lock:
                    self._deploy_num -= len(model_deploy_state.servers)
                self._gc_list.append((ManageType.MODEL, model_key))
                self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="end deploy : "
                                                                                     + model_id + ":" + version)
                result = {"CODE": "SUCCESS", "ERROR_MSG": "", "MSG": "end deploy accepted"}
        return result

    async def predict(self, model_id: str, version: str, data: list) -> dict:
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
            result["ERROR_MSG"] = "all containers are currently un available and attempting to fail back"
            return result

        if model_deploy_state.max_input is not None:
            data = data[:model_deploy_state.max_input]

        deploy_info = next(model_deploy_state.cycle_iterator)
        predict_ep = deploy_info[0]
        state = deploy_info[1]
        server_name = deploy_info[2]
        server = model_deploy_state.servers[server_name]

        if model_deploy_state.model_type == ModelType.Tensorflow:
            data = {"inputs": [data]}
            if state == StateCode.AVAILABLE:
                async with Http() as http:
                    server.ref_count += 1
                    predict_result = await http.post_json(predict_ep, data)
                    if predict_result is None:
                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                msg="http request error : " + model_id + ":" + version)
                        result["CODE"] = "FAIL"
                        result["ERROR_MSG"] = "an error occur when making inspection. please check input values"
                    elif predict_result == -1:
                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                msg="connection error : " + model_id + ":" + version)
                        await self.fail_back(model_id, version, server_name)
                        result = await self.predict(model_id, version, data.get("inputs")[0])
                    else:
                        result["CODE"] = "SUCCESS"
                        result["ERROR_MSG"] = ""
                        outputs = predict_result["outputs"]
                        result["EVNT_ID"] = outputs["result"]
                        result["PRBT"] = outputs["result_1"]
                    server.ref_count -= 1
                    return result
            else:
                result = await self.predict(model_id, version, data.get("inputs")[0])
                return result

        elif model_deploy_state.model_type == ModelType.ONNX:
            if state == StateCode.AVAILABLE:
                server.ref_count += 1
                try:
                    predict_result = ray.get(predict_ep.predict.remote(data=data))
                except Exception as exc:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                            msg="can't make inference : " + exc.__str__() + model_id + ":" + version)
                    await self.fail_back(model_id, version, server_name)
                    result = await self.predict(model_id, version, data)
                else:
                    result = predict_result
                    server.ref_count -= 1
                return result
            else:
                result = await self.predict(model_id, version, data)
                return result

    async def deploy_server(self, model_id: str, version: str, deploy_num: int,
                            model_deploy_state, model_type: int) -> dict:
        if self._deploy_num + deploy_num > self._MAX_DEPLOY:
            result = {"CODE": "FAIL", "ERROR_MSG": "max inference server exceeded",
                      "MSG": {"current container": self._deploy_num,
                              "max container number": self._MAX_DEPLOY}}
            return result
        async with self._lock:
            self._deploy_num += deploy_num
        model_key = model_id + "_" + version
        if model_type == ModelType.Tensorflow:
            # deploy_path = self._DEPLOY_PATH + model_key + "/"
            deploy_path = ROOT_DIR + self._DEPLOY_PATH + model_key + "/" + model_id
            model_path = deploy_path + "/" + str(version_encode(version))
            if os.path.isdir(model_path):
                if not any(file_name.endswith('.pb') for file_name in os.listdir(model_path)):
                    self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                            msg="model not exist:" + model_id + ":" + version)
                    result = {"CODE": "FAIL", "ERROR_MSG": "model not exist",
                              "MSG": ""}
                    async with self._lock:
                        self._deploy_num -= deploy_num
                    return result
            else:
                self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                        msg="model not exist:" + model_id + ":" + version)
                result = {"CODE": "FAIL", "ERROR_MSG": "model not exist",
                          "MSG": ""}
                async with self._lock:
                    self._deploy_num -= deploy_num
                return result

            futures = []
            list_container_name = []
            list_http_url = []
            list_grpc_url = []
            loop = asyncio.get_event_loop()
            for _ in range(deploy_num):
                http_port = self.get_port_http()
                grpc_port = self.get_port_grpc()
                name = model_id + "-" + uuid.uuid4().hex
                futures.append(
                    loop.run_in_executor(self._executor,
                                         functools.partial(self.run_container,
                                                           model_id=model_id,
                                                           container_name=name,
                                                           http_port=http_port,
                                                           grpc_port=grpc_port,
                                                           deploy_path=deploy_path)))
                list_container_name.append(name)
                list_http_url.append((self._CONTAINER_SERVER_IP, http_port))
                list_grpc_url.append((self._CONTAINER_SERVER_IP, grpc_port))
            list_container = await asyncio.gather(*futures)
            deploy_count = 0
            max_input = None
            for i in range(len(list_container)):
                if list_container[i] is not None:
                    url = "http://" + list_http_url[i][0] + ':' + str(list_http_url[i][1]) + "/v1/models/" + model_id
                    check_rst = await self.check_container_state_url(url=url)
                    if check_rst == 0:
                        if max_input is None:
                            url = "http://" + list_http_url[i][0] + ':' + str(
                                list_http_url[i][1]) + "/v1/models/" + model_id + "/metadata"
                            async with Http() as http:
                                result = await http.get(url)
                                if result is None or result == -1:
                                    self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                                            msg="can find models metadata: connection error : "
                                                                + model_id + ":" + version)
                                else:
                                    try:
                                        input_spec = result.get("metadata").get("signature_def").get("signature_def") \
                                            .get("serving_default").get("inputs")
                                    except Exception:
                                        self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                                                msg="can find models input signature : "
                                                                    + model_id + ":" + version)
                                    else:
                                        for key in input_spec:
                                            split_key = key.split("_")
                                            if split_key[0] == "seq":
                                                max_input = int(split_key[1])
                        serving_container = InferenceServer(name=list_container_name[i],
                                                            inference_server=list_container[i],
                                                            http_url=list_http_url[i], grpc_url=list_grpc_url[i],
                                                            state=StateCode.AVAILABLE)
                        model_deploy_state.servers[list_container_name[i]] = serving_container
                        deploy_count += 1
                else:
                    container = list_container[i]
                    container.remove(force=True)
                    self.release_port_http(list_http_url[i][1])
                    self.release_port_grpc(list_grpc_url[i][1])

            if deploy_count == deploy_num:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="deploy finished. " + str(deploy_count) + "/" + str(deploy_num)
                                            + " is deployed: " + model_id + ":" + version)
                async with self._lock:
                    model_deploy_state.max_input = max_input
                self._deploy_states[model_key] = model_deploy_state
                await self._set_cycle(model_id=model_id, version=version)
                result = {"CODE": "SUCCESS", "ERROR_MSG": "",
                          "MSG": "deploy finished. " + str(deploy_count) + "/" + str(deploy_num) + " deployed"}
                return result
            else:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="deploy finished. " + str(deploy_count) + "/" + str(deploy_num)
                                            + " is deployed: " + model_id + ":" + version)
                result = {"CODE": "FAIL", "ERROR_MSG": "can create container", "MSG": ""}
                async with self._lock:
                    self._deploy_num -= deploy_num
                return result

        elif model_type == ModelType.ONNX:
            actors = []
            actor_names = []
            for _ in range(deploy_num):
                name = model_id + "-" + uuid.uuid4().hex
                onnx_server = OnnxServing.options(name=name, max_concurrency=500).remote()
                actor_names.append(name)
                actors.append(onnx_server)
            init_result = []
            for actor in actors:
                res = await actor.init.remote(model_id=model_id, version=version)
                init_result.append(res)
            # init_result = ray.get([onx_serv.init.remote(model_id=model_id, version=version)
            #                                for onx_serv in actors])
            deploy_count = 0
            error_msg = ""
            for i, res in enumerate(init_result):
                code = res[0]
                msg = res[1]
                if code == -1:
                    error_msg = msg
                else:
                    serving_actor = InferenceServer(name=actor_names[i], inference_server=actors[i],
                                                    state=StateCode.AVAILABLE)
                    model_deploy_state.servers[actor_names[i]] = serving_actor
                    deploy_count += 1

            if deploy_count != deploy_num:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="can create serving actor : " + error_msg + ":" + model_id + ":" + version)
                result = {"CODE": "FAIL", "ERROR_MSG": "can create serving actor: " + error_msg, "MSG": ""}
                for actor in actors:
                    ray.kill(actor)
                async with self._lock:
                    self._deploy_num -= deploy_num
                return result
            else:
                self._deploy_states[model_key] = model_deploy_state
                await self._set_cycle(model_id=model_id, version=version)
                result = {"CODE": "SUCCESS", "ERROR_MSG": "",
                          "MSG": "deploy finished. " + str(deploy_count) + "/" + str(deploy_num) + " deployed"}
                return result

        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="unknown model type:" + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "unknown model type", "MSG": ""}
            return result

    def run_container(self, model_id: str, container_name: str, http_port: int, grpc_port: int, deploy_path: str) \
            -> Container | None:
        try:
            container = self._client.containers.run(image=self._CONTAINER_IMAGE, detach=True, name=container_name,
                                                    ports={'8501/tcp': http_port, '8500/tcp': grpc_port},
                                                    volumes=[deploy_path + ":/models/" + model_id],
                                                    environment=["MODEL_NAME=" + model_id]
                                                    )
            return container
        except ContainerError:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="the container exits with a non-zero exit code and detach is False")
            return None
        except ImageNotFound:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="tensorflow serving image does not exist")
            return None
        except APIError as e:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="the docker server returns an error : " + e.__str__())
            return None

    async def fail_back(self, model_id: str, version: str, server_name: str) -> None:
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="trying to fail back container : " + model_id + ":" + version)
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if model_deploy_state is not None:
            if model_deploy_state.state != StateCode.SHUTDOWN and model_deploy_state.state != StateCode.UNAVAILABLE:
                server = model_deploy_state.servers.get(server_name)
                server.state = StateCode.SHUTDOWN
                container_total = len(model_deploy_state.servers)
                unavailable_container = 0
                for container_key, container in model_deploy_state.servers.items():
                    if container.state == StateCode.SHUTDOWN:
                        unavailable_container += 1
                if container_total == unavailable_container:
                    model_deploy_state.state = StateCode.UNAVAILABLE
                result = await self._set_cycle(model_id, version)
                if result == 0:
                    self._gc_list.append((ManageType.SERVER, server_name))
                result_add_server = await self.add_server(model_id, version, 1)
                if result_add_server.get("CODE") == "SUCCESS":
                    model_deploy_state.state = StateCode.AVAILABLE

    async def check_container_state(self, model_id: str, version: str, container_name: str,
                                    retry_count: Optional[int] = 3):
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if model_deploy_state is not None:
            container = model_deploy_state.servers.get(container_name)
            if container.state == StateCode.AVAILABLE:
                for _ in range(retry_count):
                    async with Http() as http:
                        url = "http://" + container.http_url[0] + ':' + str(
                            container.http_url[1]) + "/v1/models/" + model_id
                        result = await http.get(url)
                        if result is None or result == -1:
                            await asyncio.sleep(0.1)
                            continue
                        else:
                            container_state_info = result["model_version_status"][0]
                            if container_state_info["state"] == "AVAILABLE":
                                return 0
                return -1

    async def check_container_state_url(self, url: str):
        for _ in range(self._RETRY_COUNT):
            async with Http() as http:
                result = await http.get(url)
                if result is None or result == -1:
                    await asyncio.sleep(self._RETRY_WAIT_TIME)
                    continue
                else:
                    container_state_info = result["model_version_status"][0]
                    if container_state_info["state"] == "AVAILABLE":
                        return 0
        return -1

    def get_port_http(self) -> int:
        port = self._http_port.pop()
        self._http_port_use.append(port)
        return port

    def get_port_grpc(self) -> int:
        port = self._grpc_port.pop()
        self._grpc_port_use.append(port)
        return port

    def release_port_http(self, port: int) -> None:
        self._http_port_use.remove(port)
        self._http_port.append(port)

    def release_port_grpc(self, port: int) -> None:
        self._grpc_port_use.remove(port)
        self._grpc_port.append(port)

    # def reset_version_config(self, host: str, name: str, base_path: str,
    #                          model_platform: str, model_version_policy: list):
    #     channel = grpc.insecure_channel(host)
    #     stub = model_service_pb2_grpc.ModelServiceStub(channel)
    #     request = model_management_pb2.ReloadConfigRequest()
    #     model_server_config = model_server_config_pb2.ModelServerConfig()
    #
    #     config_list = model_server_config_pb2.ModelConfigList()
    #     config = config_list.config.add()
    #     config.name = name
    #     config.base_path = base_path
    #     config.model_platform = model_platform
    #     for i in model_version_policy:
    #         version = version_encode(i)
    #         config.model_version_policy.specific.versions.append(version)
    #     model_server_config.model_config_list.CopyFrom(config_list)
    #     request.config.CopyFrom(model_server_config)
    #     try:
    #         response = stub.HandleReloadConfigRequest(request, 20)
    #     except grpc.RpcError as rpc_error:
    #         if rpc_error.code() == grpc.StateCode.CANCELLED:
    #             self._logger.log.remote(level=logging.ERROR, worker=self._worker,
    #                                     msg="grpc cancelled : " + name)
    #         elif rpc_error.code() == grpc.StateCode.UNAVAILABLE:
    #             self._logger.log.remote(level=logging.ERROR, worker=self._worker,
    #                                     msg="grpc unavailable : " + name)
    #         else:
    #             self._logger.log.remote(level=logging.ERROR, worker=self._worker,
    #                                     msg="grpc unknown error : " + name)
    #         return -1
    #     else:
    #         if response.status.error_code == 0:
    #             self._logger.log.remote(level=logging.INFO, worker=self._worker,
    #                                     msg="reset version config success : " + name)
    #             return 0
    #         else:
    #             self._logger.log.remote(level=logging.INFO, worker=self._worker,
    #                                     msg="reset version failed : " + response.status.error_message)
    #             return -1
    #
    # async def add_version(self, model_id: str, version: str):
    #     model_deploy_state = self._deploy_states.get(model_id + "_" + version)
    #     if model_deploy_state is None:
    #         return -1
    #
    #     if model_deploy_state.state == StateCode.AVAILABLE:
    #         result = self.copy_to_deploy(model_id, version_encode(version))
    #         if result == -1:
    #             return -1
    #         for container_name, serving_container in model_deploy_state.containers.items():
    #             result = self.reset_version_config(
    #                 host=serving_container.grpc_url[0] + ":" + str(serving_container.grpc_url[1]),
    #                 name=serving_container.name,
    #                 base_path="/models/" + model_id, model_platform="tensorflow",
    #                 model_version_policy=[version_encode(version)])
    #             if result != 0:
    #                 self._logger.log.remote(level=logging.ERROR, worker=self._worker,
    #                                         msg="an error occur when reset config :" + model_id)
    #         return 0
    #     else:
    #         self._logger.log.remote(level=logging.WARN, worker=self._worker,
    #                                 msg="the model is not usable currently : " + model_id)
    #         return -1
    #
    # def copy_to_deploy(self, model_id: str, version: int) -> int:
    #     model_path = self._project_path + "/saved_models/" + model_id + "/" + str(version)
    #     decoded_version = version_decode(version)
    #     deploy_key = model_id + "_" + decoded_version
    #     deploy_path = self._project_path + "/deploy/" + deploy_key + "/" + model_id + "/" + str(version)
    #     try:
    #         copytree(model_path, deploy_path)
    #     except FileExistsError:
    #         self._logger.log.remote(level=logging.WARN, worker=self._worker,
    #                                 msg="model file already exist in deploy dir, attempt to overwrite" + model_id + ":" + decoded_version)
    #         result = self.delete_deployed_model(model_id, version)
    #         if result == 0:
    #             self.copy_to_deploy(model_id, version)
    #     except FileNotFoundError:
    #         self._logger.log.remote(level=logging.ERROR, worker=self._worker,
    #                                 msg="model not exist" + model_id + ":" + decoded_version)
    #         return -1
    #     except shutil.Error as err:
    #         src, dist, msg = err
    #         self._logger.log.remote(level=logging.ERROR, worker=self._worker,
    #                                 msg="an error occur when copy model file. "
    #                                     "src:" + src + " target: " + dist + " | error: " + msg)
    #         return -2
    #     else:
    #         return 0
    #
    # def delete_deployed_model(self, model_id: str, version: int) -> int:
    #     decoded_version = version_decode(version)
    #     deploy_key = model_id + "_" + decoded_version
    #     deploy_path = self._project_path + "/deploy/" + deploy_key
    #     try:
    #         rmtree(deploy_path, ignore_errors=False)
    #     except shutil.Error as err:
    #         src, dist, msg = err
    #         self._logger.log.remote(level=logging.ERROR, worker=self._worker,
    #                                 msg="an error occur when delete deployed model file. "
    #                                     "src:" + src + " target: " + dist + " | error: " + msg)
    #         return -1
    #     else:
    #         return 0

    async def _set_cycle(self, model_id: str, version: str) -> int:
        cycle_list = []
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        servers = model_deploy_state.servers
        if model_deploy_state.model_type == ModelType.Tensorflow:
            async with self._lock:
                for key in servers:
                    inference_server = servers[key]
                    url = "http://" + inference_server.http_url[0] + ':' + str(inference_server.http_url[1]) + \
                          "/v1/models/" + model_id + ":predict"
                    cycle_list.append([url, inference_server.state, inference_server.name])
                model_deploy_state.cycle_iterator = cycle(cycle_list)
            return 0
        elif model_deploy_state.model_type == ModelType.ONNX:
            for key in servers:
                inference_server = servers[key]
                cycle_list.append([inference_server.inference_server, inference_server.state, inference_server.name])
            model_deploy_state.cycle_iterator = cycle(cycle_list)
            return 0

    async def gc_container(self) -> None:
        self._logger.log.remote(level=logging.DEBUG, worker=self._worker, msg="GC activate")
        for type_key in self._gc_list.copy():
            manage_type = type_key[0]
            key = type_key[1]
            if manage_type == ManageType.MODEL:
                model_deploy_state = self._deploy_states.get(key)
                servers = model_deploy_state.servers
                model_ref_count = 0
                for k, server in servers.items():
                    model_ref_count += server.ref_count
                if model_ref_count == 0:
                    remove_count = 0
                    for k, server in servers.items():
                        inference_server = server.inference_server
                        if model_deploy_state.model_type == ModelType.Tensorflow:
                            try:
                                inference_server.remove(force=True)
                            except APIError as e:
                                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                        msg="an error occur when remove container : " + e.__str__())
                                continue
                            http_port = server.http_url[1]
                            grpc_port = server.grpc_url[1]
                            self.release_port_http(http_port)
                            self.release_port_grpc(grpc_port)
                            remove_count += 1
                        elif model_deploy_state.model_type == ModelType.ONNX:
                            try:
                                ray.kill(inference_server)
                            except Exception as exc:
                                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                        msg="an error occur when remove actor : " + exc.__str__())
                                continue
                            remove_count += 1
                    del self._deploy_states[key]
                    self._gc_list.remove(type_key)
                    self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                            msg="model deploy ended: " + key)
            elif manage_type == ManageType.SERVER:
                for mds_key in self._deploy_states:
                    model_deploy_state = self._deploy_states[mds_key]
                    servers = model_deploy_state.servers
                    if key in servers:
                        sep = mds_key.split("_")
                        model_id = sep[0]
                        version = sep[1]
                        if len(servers) > 1:
                            server = servers.get(key)
                            if server.ref_count <= 0:
                                if model_deploy_state.model_type == ModelType.Tensorflow:
                                    try:
                                        server.inference_server.remove(force=True)
                                    except APIError as e:
                                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                                msg="an error occur when remove container : " + e.__str__())
                                        continue
                                    del servers[key]
                                    http_port = server.http_url[1]
                                    grpc_port = server.grpc_url[1]
                                    self.release_port_http(http_port)
                                    self.release_port_grpc(grpc_port)
                                elif model_deploy_state.model_type == ModelType.ONNX:
                                    try:
                                        ray.kill(server.inference_server)
                                    except Exception as exc:
                                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                                msg="an error occur when remove actor : " + exc.__str__())
                                        continue
                                    del servers[key]
                                await self._set_cycle(model_id, version)
                                self._gc_list.remove(type_key)
                                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                                        msg="container deleted : " + key)
                        else:
                            await self.end_deploy(model_id, version)
                            self._gc_list.remove(type_key)
                    else:
                        continue

    def get_container_list(self):
        return self._client.containers.list()

    def get_container_names(self):
        containers = self.get_container_list()
        containers = [container.name for container in containers]
        return containers


@dataclass
class InferenceServer:
    name: str
    inference_server: Container | ray.actor.ActorClass
    state: int
    ref_count: int = 0
    http_url: tuple[str, int] = field(default_factory=tuple)
    grpc_url: tuple[str, int] = field(default_factory=tuple)


@dataclass
class ModelDeployState:
    model: tuple[str, int]
    state: int
    model_type: int
    max_input: int = None
    cycle_iterator = None
    servers: dict = field(default_factory=dict)


class StateCode:
    ALREADY_EXIST = 1
    IN_PROGRESS = 3
    AVAILABLE = 0
    UNAVAILABLE = -1
    SHUTDOWN = 4


class ManageType:
    MODEL = 0
    SERVER = 1
