import json
import os
import asyncio
import shutil
import uuid
import functools
import datetime
import logging
from dataclasses import dataclass, field
from shutil import copytree, rmtree
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

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

HTTP_PORT_START = 8500
GRPC_PORT_START = 8000
MAX_CONTAINER = 20
CHECK_INTERVAL = 10
DEPLOY_PATH = ""
SERVING_CONTAINER = "tensorflow/serving:2.6.5"


@ray.remote
class ModelServing:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor()
        self._logger = ray.get_actor("logging_service")
        self._worker = type(self).__name__
        self._deploy_requests: list[tuple[str, str]] = []  # tuple of model_id and version
        self._deploy_states: dict[str, ModelDeployState] = {}
        self._gc_list: list[tuple[int, str]] = []  # tuple of manage type and key
        self._client = None
        self._http_port: list[int] = []
        self._http_port_use: list[int] = []
        self._grpc_port: list[int] = []
        self._grpc_port_use: list[int] = []
        self._ip_container_server: str = ""
        self._deploy_path: str = ""
        self._current_container_num: int = 0
        self._project_path = os.path.dirname(os.path.abspath(__file__))
        self._manager_handle = None

    async def deploy(self, model_id: str, version: str, container_num: int) -> json:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="deploy start : " + model_id
                                                                             + ":" + version)
        if (model_id, version) in self._deploy_requests:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="same deploy request is in progress : "
                                                                                 + model_id + ":" + version)
            result = json.dumps({"code": 5, "msg": "same deploy request is in progress",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            return result

        self._deploy_requests.append((model_id, version))
        encoded_version = self.version_encode(version)
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is not None:
            if model_deploy_state.state == StateCode.AVAILABLE:
                deploy_num = len(model_deploy_state.containers)
                diff = container_num - deploy_num
                if diff > 0:
                    result = await self.add_container(model_id, version, diff)
                    return result
                elif diff < 0:
                    self._deploy_requests.remove((model_id, version))
                    result = await self.remove_container(model_id, version, abs(diff))
                    return result
                else:
                    result = json.dumps({"code": 4, "msg": "nothing to change",
                                         "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                        'utf8')
                    return result

        if container_num <= 0:
            self._deploy_requests.remove((model_id, version))
            result = json.dumps({"code": 4, "msg": "nothing to change",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                'utf8')
            return result

        if (self._current_container_num + container_num) > MAX_CONTAINER:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="max container number exceeded : "
                                                                                 + model_id + ":" + version)
            result = json.dumps({"code": 4, "msg": "max container number exceeded",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                'utf8')
            return result

        cp_result = self.copy_to_deploy(model_id, encoded_version)
        if cp_result == -1:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when copying model file :" + model_id + ":" + version)
            result = json.dumps({"code": 7, "msg": "an error occur when copying model file",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                'utf8')
            async with self._lock:
                self._current_container_num -= container_num
            return result

        model_deploy_state = ModelDeployState(model=(model_id, encoded_version), state=StateCode.AVAILABLE)
        result = await self.deploy_containers(model_id, version, container_num, model_deploy_state)
        return result

    async def get_deploy_state(self) -> json:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get deploy state")
        deploy_states = []
        for key in self._deploy_states:
            sep_key = key.split("_")
            model_id = sep_key[0]
            version = sep_key[1]
            model_deploy_state = self._deploy_states[key]
            container_num = len(model_deploy_state.containers)
            deploy_state = {"model_id": model_id, "version": version, "container_num": container_num}
            deploy_states.append(deploy_state)
        return json.dumps({"deploy_states": deploy_states,
                           "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')

    async def add_container(self, model_id: str, version: str, container_num: int) -> json:
        if (self._current_container_num + container_num) > MAX_CONTAINER:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="max container number exceeded : "
                                                                                 + model_id + ":" + version)
            result = json.dumps({"code": 4, "msg": "max container number exceeded",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="the model is not deployed : "
                                                                                 + model_id + ":" + version)
            result = json.dumps({"code": 8, "msg": "the model is not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        result = await self.deploy_containers(model_id, version, container_num, model_deploy_state)
        return result

    async def remove_container(self, model_id: str, version: str, container_num: int) -> json:
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="the model is not deployed : "
                                                                                 + model_id + ":" + version)
            result = json.dumps({"code": 8, "msg": "the model is not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        if container_num >= len(model_deploy_state.containers):
            result = await self.end_deploy(model_id, version)
            return result

        containers = model_deploy_state.containers
        gc_list = []
        for i, key in enumerate(containers):
            if i < container_num:
                container = containers[key]
                container.state = StateCode.SHUTDOWN
                gc_list.append((ManageType.CONTAINER, key))
            else:
                break

        result = await self._set_cycle(model_id, version)
        if result == 0:
            self._gc_list = self._gc_list + gc_list
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="shutdown pending on " + str(container_num) + " containers : " + model_id + ":"
                                        + version)
            result = json.dumps({"code": 200, "msg": "shutdown pending on " + str(container_num) + " containers",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

    async def end_deploy(self, model_id: str, version: str) -> json:
        encoded_version = self.version_encode(version)
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="model not deployed : "
                                                                                 + model_id + ":" + version)
            result = json.dumps({"code": 8, "msg": "model not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        model_deploy_state.state = StateCode.SHUTDOWN
        self._gc_list.append((ManageType.MODEL, model_key))
        self.delete_deployed_model(model_id, encoded_version)
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="end deploy : "
                                                                             + model_id + ":" + version)
        result = json.dumps({"code": 200, "msg": "end_deploy accepted",
                             "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
        return result

    async def predict(self, model_id: str, version: str, data: dict):
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if (model_deploy_state is None) or (model_deploy_state.cycle_iterator is None):
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="model not deployed : "
                                                                                 + model_id + ":" + version)
            result = json.dumps({"code": 10, "msg": "the model not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        if model_deploy_state.state == StateCode.SHUTDOWN:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="model not deployed : "
                                                                                 + model_id + ":" + version)
            result = json.dumps({"code": 10, "msg": "the model not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        deploy_info = next(model_deploy_state.cycle_iterator)
        url = deploy_info[0]
        state = deploy_info[1]
        container_name = deploy_info[2]
        container = model_deploy_state.containers[container_name]
        if state == StateCode.AVAILABLE:
            async with Http() as http:
                async with self._lock:
                    container.ref_count += 1
                result = await http.post_json(url, data)
                if result is None:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg="http request error : "
                                                                                          + model_id + ":" + version)
                    result = json.dumps({"code": 10, "msg": "http request error"}).encode('utf8')
                elif result == -1:
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg="connection error : "
                                                                                          + model_id + ":" + version)
                    result = json.dumps({"code": 11, "msg": "connection error trying to fail back"}).encode('utf8')
                    await self.fail_back(model_id, version, container_name)
                    result = await self.predict(model_id, version, data)
                async with self._lock:
                    container.ref_count -= 1
                return result
        else:
            result = await self.predict(model_id, version, data)
            return result

    async def deploy_containers(self, model_id: str, version: str, container_num: int, model_deploy_state):
        async with self._lock:
            self._current_container_num += container_num
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
                                                       deploy_path=self._deploy_path + model_key + "/")))
            list_container_name.append(container_name)
            list_http_url.append((self._ip_container_server, http_port))
            list_grpc_url.append((self._ip_container_server, grpc_port))
        list_container = await asyncio.gather(*futures)
        deploy_count = 0
        for i in range(len(list_container)):
            if list_container[i] is not None:
                serving_container = ServingContainer(name=list_container_name[i], container=list_container[i],
                                                     http_url=list_http_url[i], grpc_url=list_grpc_url[i],
                                                     state=StateCode.AVAILABLE)
                model_deploy_state.containers[list_container_name[i]] = serving_container
                deploy_count += 1
        async with self._lock:
            self._current_container_num -= (container_num - deploy_count)
        self._deploy_states[model_key] = model_deploy_state
        self._deploy_requests.remove((model_id, version))
        await self._set_cycle(model_id=model_id, version=version)
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="deploy finished. " + str(deploy_count) + "/" + str(container_num)
                                    + " is deployed: " + model_id + ":" + version)
        result = json.dumps({"code": 0,
                             "msg": "deploy finished. " + str(deploy_count) + "/" + str(container_num) + "deployed",
                             "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
        return result

    def run_container(self, model_id: str, container_name: str, http_port: int, grpc_port: int, deploy_path: str):
        try:
            container = self._client.containers.run(image=SERVING_CONTAINER, detach=True, name=container_name,
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

    async def fail_back(self, model_id: str, version: str, container_name: str):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="trying to fail back container : " + model_id + ":" + version)
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if model_deploy_state is not None:
            container = model_deploy_state.containers.get(container_name)
            container.state = StateCode.SHUTDOWN
            result = await self._set_cycle(model_id, version)
            if result == 0:
                self._gc_list.append((ManageType.CONTAINER, container_name))
            await self.add_container(model_id, version, 1)

    async def get_model_state(self, model_name: str):
        url = self.get_model_endpoint(model_name)
        async with Http() as http:
            result = await http.get(url)
            result = result.decode('utf8').replace("'", '"')
            return result

    def init(self) -> int:
        """
        Initialize serving process
        :return: success 0 / fail -1
        """
        # set param from config file
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="init process : set config...")
        self._ip_container_server = "localhost"  # config
        self._deploy_path = os.path.dirname(os.path.abspath(__file__)) + "/deploy/"  # config

        # make port list
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="init process : set port...")
        for i in range(MAX_CONTAINER * 2):
            self._grpc_port.append(GRPC_PORT_START + i)
            self._http_port.append(HTTP_PORT_START + i)

        # set docker client
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="init process : init docker client...")
        try:
            self._client = docker.from_env()
        except DockerException as e:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="can't make connection to docker client" + e.__str__())
            return -1

        # set GC
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="init process : init GC...")
        self._manager_handle = AsyncIOScheduler()
        self._manager_handle.add_job(self.gc_container, "interval", seconds=CHECK_INTERVAL, id="gc_container")
        self._manager_handle.start()
        return 0

    def get_port_http(self):
        port = self._http_port.pop()
        self._http_port_use.append(port)
        return port

    def get_port_grpc(self):
        port = self._grpc_port.pop()
        self._grpc_port_use.append(port)
        return port

    def release_port_http(self, port: int):
        self._http_port_use.remove(port)
        self._http_port.append(port)

    def release_port_grpc(self, port: int):
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
            version = self.version_encode(i)
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
            result = self.copy_to_deploy(model_id, self.version_encode(version))
            if result == -1:
                return -1
            for container_name, serving_container in model_deploy_state.containers.items():
                result = self.reset_version_config(host=serving_container.grpc_url[0] +":" + str(serving_container.grpc_url[1]),
                                                   name=serving_container.name,
                                                   base_path="/models/" + model_id, model_platform="tensorflow",
                                                   model_version_policy=[self.version_encode(version)])
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
        decoded_version = self.version_decode(version)
        deploy_key = model_id + "_" + decoded_version
        deploy_path = self._project_path + "/deploy/" + deploy_key + "/" + model_id + "/" + str(version)
        try:
            copytree(model_path, deploy_path)
        except FileExistsError:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="model file already exist in deploy dir" + model_id + ":" + decoded_version)
            return 1
        except shutil.Error as err:
            src, dist, msg = err
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when copy model file. "
                                        "src:" + src + " target: " + dist + " | error: " + msg)
            return -1
        else:
            return 0

    def delete_deployed_model(self, model_id: str, version: int) -> int:
        decoded_version = self.version_decode(version)
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

    async def gc_container(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="GC activate")
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
                    del self._deploy_states[key]
                    self._gc_list.remove(type_key)
                    async with self._lock:
                        self._current_container_num -= remove_count
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
                                    self._current_container_num -= 1
                                http_port = container.http_url[1]
                                grpc_port = container.grpc_url[1]
                                self.release_port_http(http_port)
                                self.release_port_grpc(grpc_port)
                                del containers[key]
                                await self._set_cycle(model_id, version)
                                self._gc_list.remove(type_key)
                                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                                        msg="container deleted : " + key)
                        else:
                            await self.end_deploy(model_id, version)
                            self._gc_list.remove(type_key)
                    else:
                        continue
        print(self._deploy_states)

    def version_encode(self, version: str) -> int:
        sv = version.split('.')
        if len(sv[-1]) > 9:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg="can't exceed decimal point over 9")
            return -1
        else:
            encoded = sv[0] + sv[-1] + str(len(sv[-1]))
            return int(encoded)

    def version_decode(self, version: int):
        decimal = version // 10 ** 0 % 10
        decoded = str(version)[:-1]
        decimal = len(decoded) - decimal
        if decimal > 0:
            decoded = decoded[:decimal] + "." + decoded[decimal:]
        return decoded

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
    cycle_iterator = None
    containers: dict = field(default_factory=dict)


class StateCode:
    ALREADY_EXIST = 1
    IN_PROGRESS = 3
    AVAILABLE = 0
    SHUTDOWN = 4


class ManageType:
    MODEL = 0
    CONTAINER = 1
