import configparser
import os
import asyncio
import shutil
import uuid
import functools
import logging
import logger
from dataclasses import dataclass, field
from shutil import copytree, rmtree
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
from statics import Actors
from db import DBUtil


@ray.remote
class ModelServing:
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
    _current_container_num : int
        A number of deployed container currently
    _project_path : str
        Absolute path of project
    _manager_handle : AsyncIOScheduler
        An AsyncIOScheduler instance for managing container
    _HTTP_PORT_START : int
        Configuration of http port range
    _GRPC_PORT_START : int
        Configuration of grpc port range
    _CONTAINER_MAX : int
        Configuration of max container number
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
    copy_to_deploy(model_id: str, version: int) -> int:
        Copy saved model file to deploy directory
    delete_deployed_model(model_id: str, version: int) -> int:
        Delete saved model file from deploy directory
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
        self._current_container_num: int = 0
        self._project_path: str = os.path.dirname(os.path.abspath(__file__))
        self._manager_handle: AsyncIOScheduler | None = None
        self._HTTP_PORT_START: int = 8500
        self._GRPC_PORT_START: int = 8000
        self._CONTAINER_MAX: int = 20
        self._GC_CHECK_INTERVAL: int = 10
        self._DEPLOY_PATH: str = ''
        self._CONTAINER_IMAGE: str = "tensorflow/serving:2.6.5"
        self._CONTAINER_SERVER_IP: str = ""
        self._RETRY_COUNT: int = 20
        self._RETRY_WAIT_TIME: float = 0.1

    async def init(self) -> int:
        self._boot_logger.info("(" + self._worker + ") " + "init model_serving actor...")
        self._boot_logger.info("(" + self._worker + ") " + "set global logger...")
        self._logger = ray.get_actor(Actors.LOGGER)
        self._boot_logger.info("(" + self._worker + ") " + "set statics from config...")
        config_parser = configparser.ConfigParser()
        try:
            config_parser.read("config/config.ini")
            self._HTTP_PORT_START = int(config_parser.get("DEPLOY", "HTTP_PORT_START"))
            self._GRPC_PORT_START = int(config_parser.get("DEPLOY", "GRPC_PORT_START"))
            self._CONTAINER_MAX = int(config_parser.get("DEPLOY", "CONTAINER_MAX"))
            self._GC_CHECK_INTERVAL = int(config_parser.get("DEPLOY", "GC_CHECK_INTERVAL"))
            self._DEPLOY_PATH = str(config_parser.get("DEPLOY", "DEPLOY_PATH"))
            self._CONTAINER_IMAGE = str(config_parser.get("DEPLOY", "CONTAINER_IMAGE"))
            self._CONTAINER_SERVER_IP = str(config_parser.get("DEPLOY", "CONTAINER_SERVER_IP"))
            self._RETRY_COUNT: int = int(config_parser.get("DEPLOY", "RETRY_COUNT"))
            self._RETRY_WAIT_TIME: float = float(config_parser.get("DEPLOY", "RETRY_WAIT_TIME"))
        except configparser.Error as e:
            self._boot_logger.error("(" + self._worker + ") " + "an error occur when set config...: " + str(e))
            return -1

        self._boot_logger.info("(" + self._worker + ") " + "set container port range...")
        for i in range(self._CONTAINER_MAX * 2):
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

        self._boot_logger.info("(" + self._worker + ") " + "deploying existing containers...")
        self._db = DBUtil()
        stored_deploy_states = []
        try:
            stored_deploy_states = self._db.select(name="select_deploy_state")
        except Exception as exc:
            self._boot_logger.error(
                "(" + self._worker + ") " + "can't read deploy state from db:" + exc.__str__())
        for stored_deploy_state in stored_deploy_states:
            model_id = stored_deploy_state[0]
            mn_ver = str(stored_deploy_state[1])
            n_ver = str(stored_deploy_state[2])
            container_num = stored_deploy_state[3]
            print(model_id, mn_ver, n_ver, container_num)
            version = mn_ver+"."+n_ver
            encoded_version = version_encode(version)
            model_deploy_state = ModelDeployState(model=(model_id, encoded_version),
                                                  state=StateCode.AVAILABLE)
            result = await self.deploy_containers(model_id, version, container_num, model_deploy_state)
            if result.get("CODE") == "FAIL":
                self._boot_logger.error(
                    "(" + self._worker + ") " + "can't make container from stored deploy state")
                return -1
        self._boot_logger.info("(" + self._worker + ") " + "init model_serving actor complete...")
        return 0

    async def deploy(self, model_id: str, version: str, container_num: int) -> dict:
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
                diff = container_num - self._current_container_num
                if diff > 0:
                    try:
                        result = await self.add_container(model_id, version, diff)
                    except Exception as exc:
                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                msg="deploy error : " + model_id
                                                    + ":" + version + ":" + exc.__str__())
                elif diff < 0:
                    try:
                        result = await self.remove_container(model_id, version, abs(diff))
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
            if container_num <= 0:
                result = {"CODE": "SUCCESS", "ERROR_MSG": "", "MSG": "nothing to change"}
            else:
                cp_result = self.copy_to_deploy(model_id, encoded_version)
                if cp_result == -1:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                            msg="an error occur when copying model file :" + model_id + ":" + version)
                    result = {"CODE": "FAIL", "ERROR_MSG": "model not found", "MSG": ""}
                elif cp_result == -2:
                    result = {"CODE": "FAIL", "ERROR_MSG": "an error occur when copying model file", "MSG": ""}
                else:
                    model_deploy_state = ModelDeployState(model=(model_id, encoded_version),
                                                          state=StateCode.AVAILABLE)
                    try:
                        result = await self.deploy_containers(model_id, version, container_num, model_deploy_state)
                    except Exception as exc:
                        self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                msg="deploy error : " + model_id
                                                    + ":" + version + ":" + exc.__str__())
            self._deploy_requests.remove((model_id, version))
            return result

    async def get_deploy_state(self, model_id: str, version: str) -> dict:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get deploy state")
        model_key = model_id + "_" + version
        if model_key in self._deploy_states:
            containers = []
            sep_key = model_key.split("_")
            model_id = sep_key[0]
            version = sep_key[1]
            model_deploy_state = self._deploy_states[model_key]
            container_num = len(model_deploy_state.containers)
            for k, v in model_deploy_state.containers.items():
                containers.append((v.name, v.state))
                res = await self.check_container_state(model_id, version, container_name=v.name)  # test
                print(res)  # test
            deploy_state = {"model_id": model_id, "version": version, "containers": containers, "current_container_num":self._current_container_num}
            return {"CODE": "SUCCESS", "ERROR_MSG": "", "DEPLOY_STATE": deploy_state}
        else:
            return {"CODE": "FAIL", "ERROR_MSG": "model not found", "DEPLOY_STATE": ""}

    async def add_container(self, model_id: str, version: str, container_num: int) -> dict:
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="the model is not deployed : "
                                                                                 + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "the model is not deployed", "MSG": ""}
            return result
        result = await self.deploy_containers(model_id, version, container_num, model_deploy_state)
        return result

    def remove_all_container(self) -> None:
        containers = self._client.containers.list(filters={"ancestor": self._CONTAINER_IMAGE})
        for container in containers:
            container.remove(force=True)

    async def remove_container(self, model_id: str, version: str, container_num: int) -> dict:
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="the model is not deployed : "
                                                                                 + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "the model is not deployed", "MSG": ""}
            return result

        if model_deploy_state.state == StateCode.SHUTDOWN:
            return {"CODE": "FAIL", "ERROR_MSG": "shutdown has already been scheduled on this model", "MSG": ""}

        containers = model_deploy_state.containers
        available_containers = []
        for k, v in containers.items():
            if v.state == StateCode.AVAILABLE:
                available_containers.append(v.container)

        if container_num >= len(available_containers):
            result = await self.end_deploy(model_id, version)
            return result

        gc_list = []
        for i, key in enumerate(containers):
            if i < container_num:
                container = containers[key]
                if container.state == StateCode.AVAILABLE:
                    container.state = StateCode.SHUTDOWN
                    gc_list.append((ManageType.CONTAINER, key))
            else:
                break

        result = await self._set_cycle(model_id, version)
        if result == 0:
            self._gc_list = self._gc_list + gc_list
            async with self._lock:
                self._current_container_num -= len(gc_list)
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="shutdown pending on " + str(container_num) + " containers : " + model_id + ":"
                                        + version)
            result = {"CODE": "SUCCESS", "ERROR_MSG": "",
                      "MSG": "shutdown pending on " + str(container_num) + " containers"}
            return result
        else:
            result = {"CODE": "FAIL", "ERROR_MSG": "failed to remove container", "MSG": ""}
            return result

    async def end_deploy(self, model_id: str, version: str) -> dict:
        encoded_version = version_encode(version)
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
                async with self._lock:
                    model_deploy_state.state = StateCode.SHUTDOWN
                    self._gc_list.append((ManageType.MODEL, model_key))
                    self._current_container_num -= len(model_deploy_state.containers)
                try:
                    self.delete_deployed_model(model_id, encoded_version)
                except Exception as exc:
                    self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                            msg="end deploy : model not deleted: "
                                                + model_id + ":" + version + ":" + exc.__str__())
                finally:
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
        elif model_deploy_state.state == StateCode.UN_AVAILABLE:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="all containers are currently un available : " + model_id + ":" + version)
            result["CODE"] = "FAIL"
            result["ERROR_MSG"] = "all containers are currently un available and attempting to fail back"
            return result

        if model_deploy_state.max_input is not None:
            data = data[:model_deploy_state.max_input]

        data = {"inputs": [data]}
        deploy_info = next(model_deploy_state.cycle_iterator)
        url = deploy_info[0]
        state = deploy_info[1]
        container_name = deploy_info[2]
        container = model_deploy_state.containers[container_name]
        if state == StateCode.AVAILABLE:
            async with Http() as http:
                async with self._lock:
                    container.ref_count += 1
                predict_result = await http.post_json(url, data)
                if predict_result is None:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg="http request error : "
                                                                                          + model_id + ":" + version)
                    result["CODE"] = "FAIL"
                    result["ERROR_MSG"] = "an error occur when making inspection. please check input values"
                elif predict_result == -1:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg="connection error : "
                                                                                          + model_id + ":" + version)
                    await self.fail_back(model_id, version, container_name)
                    result = await self.predict(model_id, version, data.get("inputs")[0])
                else:
                    async with self._lock:
                        container.ref_count -= 1
                    result["CODE"] = "SUCCESS"
                    result["ERROR_MSG"] = ""
                    outputs = predict_result["outputs"]
                    result["EVNT_ID"] = outputs["result"]
                    result["PRBT"] = outputs["result_1"]
                return result
        else:
            result = await self.predict(model_id, version, data.get("inputs")[0])
            return result

    async def deploy_containers(self, model_id: str, version: str, container_num: int, model_deploy_state) -> dict:
        async with self._lock:
            if self._current_container_num + container_num > self._CONTAINER_MAX:
                result = {"CODE": "FAIL", "ERROR_MSG": "max container number exceeded",
                          "MSG": {"current container": self._current_container_num,
                                  "max container number": self._CONTAINER_MAX}}
                return result

        model_key = model_id + "_" + version
        futures = []
        list_container_name = []
        list_http_url = []
        list_grpc_url = []
        loop = asyncio.get_event_loop()
        for _ in range(container_num):
            http_port = self.get_port_http()
            grpc_port = self.get_port_grpc()
            container_name = model_id + "-" + uuid.uuid4().hex
            futures.append(
                loop.run_in_executor(self._executor,
                                     functools.partial(self.run_container,
                                                       model_id=model_id,
                                                       container_name=container_name,
                                                       http_port=http_port,
                                                       grpc_port=grpc_port,
                                                       # deploy_path=self._project_path + self._DEPLOY_PATH + model_key + "/"))) #remove pp
                                                       deploy_path=self._DEPLOY_PATH + model_key + "/")))
            list_container_name.append(container_name)
            list_http_url.append((self._CONTAINER_SERVER_IP, http_port))
            list_grpc_url.append((self._CONTAINER_SERVER_IP, grpc_port))
        list_container = await asyncio.gather(*futures)
        deploy_count = 0
        max_input = None
        for i in range(len(list_container)):
            if list_container[i] is not None:
                url = "http://"+list_http_url[i][0]+':'+str(list_http_url[i][1])+"/v1/models/"+model_id
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
                                    input_spec = result.get("metadata").get("signature_def").get("signature_def")\
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
                    if max_input is not None:
                        serving_container = ServingContainer(name=list_container_name[i], container=list_container[i],
                                                             http_url=list_http_url[i], grpc_url=list_grpc_url[i],
                                                             state=StateCode.AVAILABLE)
                        model_deploy_state.containers[list_container_name[i]] = serving_container
                        deploy_count += 1
                    else:
                        container = list_container[i]
                        container.remove(force=True)
                        self.release_port_http(list_http_url[i][1])
                        self.release_port_grpc(list_grpc_url[i][1])
            else:
                container = list_container[i]
                container.remove(force=True)
                self.release_port_http(list_http_url[i][1])
                self.release_port_grpc(list_grpc_url[i][1])

        if deploy_count == container_num:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="deploy finished. " + str(deploy_count) + "/" + str(container_num)
                                        + " is deployed: " + model_id + ":" + version)
            async with self._lock:
                model_deploy_state.max_input = max_input
                self._deploy_states[model_key] = model_deploy_state
                self._current_container_num += container_num
            await self._set_cycle(model_id=model_id, version=version)
            result = {"CODE": "SUCCESS", "ERROR_MSG": "",
                      "MSG": "deploy finished. " + str(deploy_count) + "/" + str(container_num) + " deployed"}
            return result
        else:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="deploy finished. " + str(deploy_count) + "/" + str(container_num)
                                        + " is deployed: " + model_id + ":" + version)
            result = {"CODE": "FAIL", "ERROR_MSG": "can create container", "MSG": ""}
            return result

    def run_container(self, model_id: str, container_name: str, http_port: int, grpc_port: int, deploy_path: str) \
            -> Container | None:
        try:
            container = self._client.containers.run(image=self._CONTAINER_IMAGE, detach=True, name=container_name,
                                                    ports={'8501/tcp': http_port, '8500/tcp': grpc_port},
                                                    volumes=[deploy_path + model_id + ":/models/" + model_id],
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

    async def fail_back(self, model_id: str, version: str, container_name: str) -> None:
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="trying to fail back container : " + model_id + ":" + version)
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if model_deploy_state is not None:
            if model_deploy_state.state != StateCode.SHUTDOWN and model_deploy_state.state != StateCode.UN_AVAILABLE:
                container = model_deploy_state.containers.get(container_name)
                container.state = StateCode.SHUTDOWN
                container_total = len(model_deploy_state.containers)
                un_available_container = 0
                for container_key, container in model_deploy_state.containers.items():
                    if container.state == StateCode.SHUTDOWN:
                        un_available_container += 1
                if container_total == un_available_container:
                    model_deploy_state.state = StateCode.UN_AVAILABLE
                result = await self._set_cycle(model_id, version)
                if result == 0:
                    self._gc_list.append((ManageType.CONTAINER, container_name))
                result_add_container = await self.add_container(model_id, version, 1)
                if result_add_container.get("CODE") != "SUCCESS":
                    model_deploy_state.state = StateCode.AVAILABLE

    async def check_container_state(self, model_id: str, version: str, container_name: str,
                                    retry_count: Optional[int] = 3):
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if model_deploy_state is not None:
            container = model_deploy_state.containers.get(container_name)
            if container.state == StateCode.AVAILABLE:
                for _ in range(retry_count):
                    async with Http() as http:
                        url = "http://" + container.http_url[0] + ':' + str(container.http_url[1]) + "/v1/models/" + model_id
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

    def reset_version_config(self, host: str, name: str, base_path: str,
                             model_platform: str, model_version_policy: list):
        channel = grpc.insecure_channel(host)
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        request = model_management_pb2.ReloadConfigRequest()
        model_server_config = model_server_config_pb2.ModelServerConfig()

        config_list = model_server_config_pb2.ModelConfigList()
        config = config_list.config.add()
        config.name = name
        config.base_path = base_path
        config.model_platform = model_platform
        for i in model_version_policy:
            version = version_encode(i)
            config.model_version_policy.specific.versions.append(version)
        model_server_config.model_config_list.CopyFrom(config_list)
        request.config.CopyFrom(model_server_config)
        try:
            response = stub.HandleReloadConfigRequest(request, 20)
        except grpc.RpcError as rpc_error:
            if rpc_error.code() == grpc.StateCode.CANCELLED:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="grpc cancelled : " + name)
            elif rpc_error.code() == grpc.StateCode.UNAVAILABLE:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="grpc unavailable : " + name)
            else:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="grpc unknown error : " + name)
            return -1
        else:
            if response.status.error_code == 0:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="reset version config success : " + name)
                return 0
            else:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="reset version failed : " + response.status.error_message)
                return -1

    async def add_version(self, model_id: str, version: str):
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if model_deploy_state is None:
            return -1

        if model_deploy_state.state == StateCode.AVAILABLE:
            result = self.copy_to_deploy(model_id, version_encode(version))
            if result == -1:
                return -1
            for container_name, serving_container in model_deploy_state.containers.items():
                result = self.reset_version_config(
                    host=serving_container.grpc_url[0] + ":" + str(serving_container.grpc_url[1]),
                    name=serving_container.name,
                    base_path="/models/" + model_id, model_platform="tensorflow",
                    model_version_policy=[version_encode(version)])
                if result != 0:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                            msg="an error occur when reset config :" + model_id)
            return 0
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="the model is not usable currently : " + model_id)
            return -1

    def copy_to_deploy(self, model_id: str, version: int) -> int:
        model_path = self._project_path + "/saved_models/" + model_id + "/" + str(version)
        decoded_version = version_decode(version)
        deploy_key = model_id + "_" + decoded_version
        deploy_path = self._project_path + "/deploy/" + deploy_key + "/" + model_id + "/" + str(version)
        try:
            copytree(model_path, deploy_path)
        except FileExistsError:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="model file already exist in deploy dir, attempt to overwrite" + model_id + ":" + decoded_version)
            result = self.delete_deployed_model(model_id, version)
            if result == 0:
                self.copy_to_deploy(model_id, version)
        except FileNotFoundError:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="model not exist" + model_id + ":" + decoded_version)
            return -1
        except shutil.Error as err:
            src, dist, msg = err
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when copy model file. "
                                        "src:" + src + " target: " + dist + " | error: " + msg)
            return -2
        else:
            return 0

    def delete_deployed_model(self, model_id: str, version: int) -> int:
        decoded_version = version_decode(version)
        deploy_key = model_id + "_" + decoded_version
        deploy_path = self._project_path + "/deploy/" + deploy_key
        try:
            rmtree(deploy_path, ignore_errors=False)
        except shutil.Error as err:
            src, dist, msg = err
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when delete deployed model file. "
                                        "src:" + src + " target: " + dist + " | error: " + msg)
            return -1
        else:
            return 0

    async def _set_cycle(self, model_id: str, version: str) -> int:
        cycle_list = []
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        containers = model_deploy_state.containers
        async with self._lock:
            for key in containers:
                container = containers[key]
                url = "http://" + container.http_url[0] + ':' + str(container.http_url[1]) + \
                      "/v1/models/" + model_id + ":predict"
                cycle_list.append([url, container.state, container.name])
            model_deploy_state.cycle_iterator = cycle(cycle_list)
        return 0

    async def gc_container(self) -> None:
        self._logger.log.remote(level=logging.DEBUG, worker=self._worker, msg="GC activate")
        for type_key in self._gc_list.copy():
            manage_type = type_key[0]
            key = type_key[1]
            if manage_type == ManageType.MODEL:
                model_deploy_state = self._deploy_states.get(key)
                containers = model_deploy_state.containers
                model_ref_count = 0
                for container_name, serving_container in containers.items():
                    model_ref_count += serving_container.ref_count
                if model_ref_count == 0:
                    remove_count = 0
                    for container_name, serving_container in containers.items():
                        container = serving_container.container
                        try:
                            container.remove(force=True)
                        except APIError as e:
                            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                    msg="an error occur when remove container : " + e.__str__())
                            continue
                        http_port = serving_container.http_url[1]
                        grpc_port = serving_container.grpc_url[1]
                        self.release_port_http(http_port)
                        self.release_port_grpc(grpc_port)
                        remove_count += 1
                    async with self._lock:
                        del self._deploy_states[key]
                        self._gc_list.remove(type_key)
                    self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                            msg="model deploy ended: " + key)
            elif manage_type == ManageType.CONTAINER:
                for mds_key in self._deploy_states:
                    model_deploy_state = self._deploy_states[mds_key]
                    containers = model_deploy_state.containers
                    if key in containers:
                        sep = mds_key.split("_")
                        model_id = sep[0]
                        version = sep[1]
                        if len(containers) > 1:
                            container = containers.get(key)
                            if container.ref_count <= 0:
                                try:
                                    container.container.remove(force=True)
                                except APIError as e:
                                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                                            msg="an error occur when remove container : " + e.__str__())
                                    continue
                                async with self._lock:
                                    del containers[key]
                                http_port = container.http_url[1]
                                grpc_port = container.grpc_url[1]
                                self.release_port_http(http_port)
                                self.release_port_grpc(grpc_port)
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
class ServingContainer:
    name: str
    container: Container
    state: int
    ref_count: int = 0
    http_url: tuple[str, int] = field(default_factory=tuple)
    grpc_url: tuple[str, int] = field(default_factory=tuple)


@dataclass
class ModelDeployState:
    model: tuple[str, int]
    state: int
    max_input: int = None
    cycle_iterator = None
    containers: dict = field(default_factory=dict)


class StateCode:
    ALREADY_EXIST = 1
    IN_PROGRESS = 3
    AVAILABLE = 0
    UN_AVAILABLE = -1
    SHUTDOWN = 4


class ManageType:
    MODEL = 0
    CONTAINER = 1
