import json
import os
import asyncio
import shutil
import uuid
import functools
from dataclasses import dataclass, field
from shutil import copytree, rmtree
from concurrent.futures import ThreadPoolExecutor

import docker
import ray
import grpc
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.config import model_server_config_pb2
from asyncstdlib.itertools import cycle
from docker.errors import ContainerError, ImageNotFound, APIError
from docker.models.containers import Container
# from aioshutil import copytree

from http_util import Http

HTTP_PORT_START = 8500
GRPC_PORT_START = 8000
MAX_CONTAINER = 50
DEPLOY_PATH = ""


@ray.remote
class ModelServing:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor()
        self._deploy_requests = []
        self._deploy_states = {}
        self._client = None
        self._http_port = []
        self._http_port_use = []
        self._grpc_port = []
        self._grpc_port_use = []
        self._ip_container_server = None
        self._deploy_path = None
        self._current_container_num = 0
        self._project_path = os.path.dirname(os.path.abspath(__file__))
        self.init_client()

    async def deploy(self, model_id: str, model_version: str, container_num: int) -> json:
        if (self._current_container_num + container_num) > MAX_CONTAINER:
            print("max container number exceeded")
            result = json.dumps({"code": 4, "msg": "max container number exceeded"})
            return result

        is_in_progress = False
        async with self._lock:
            if (model_id, model_version) in self._deploy_requests:
                is_in_progress = True

        if is_in_progress:
            print("same deploy request is in progress")
            result = json.dumps({"code": 5, "msg": "same deploy request is in progress"})
            return result
        else:
            async with self._lock:
                self._deploy_requests.append((model_id, model_version))
            model_deploy_state = None
            encoded_version = self.version_encode(model_version)
            model_key = model_id + ":" + model_version
            if model_key in self._deploy_states:
                model_deploy_state = self._deploy_states.get(model_key)
            if model_deploy_state.state == StatusCode.ALREADY_EXIST:
                print("model already deployed.")
                result = json.dumps({"code": 6, "msg": "model already deployed"})
                return result
            else:
                async with self._lock:
                    self._current_container_num += container_num
                result = self.copy_to_deploy(model_id, encoded_version)
                if result == -1:
                    print("an error occur when copying model file")
                    result = json.dumps({"code": 7, "msg": "n error occur when copying model file"})
                    async with self._lock:
                        self._current_container_num -= container_num
                    return result
                model_deploy_state = ModelDeployState(model=(model_id, encoded_version), state=StatusCode.ALREADY_EXIST)
                is_deploy_failed = False
                for _ in range(container_num):
                    http_port = self.get_port_http()
                    grpc_port = self.get_port_grpc()
                    container_name = model_id + "-" + uuid.uuid4().hex
                    # loop = asyncio.get_event_loop()
                    # container = await loop.run_in_executor(self._executor,
                    #                                        functools.partial(self.run_container,
                    #                                                          model_id=model_id,
                    #                                                          container_name=container_name,
                    #                                                          http_port=http_port,
                    #                                                          grpc_port=grpc_port,
                    #                                                          deploy_path=self._deploy_path + model_key + "/"))
                    container = await self.run_container(model_id=model_id, container_name=container_name,
                                                         http_port=http_port, grpc_port=grpc_port,
                                                         deploy_path=self._deploy_path+model_key+"/")
                    if container is None:
                        is_deploy_failed = True
                        break
                    serving_container = ServingContainer(name=container_name, container=container,
                                                         http_url=(self._ip_container_server, http_port),
                                                         grpc_url=(self._ip_container_server, grpc_port))
                    model_deploy_state.containers.append(serving_container)
                if is_deploy_failed:
                    for serving_container in model_deploy_state.containers:
                        serving_container.container.remove(force=True)
                    async with self._lock:
                        self._current_container_num -= container_num
                    print("failed to create container")
                    result = json.dumps({"code": 8, "msg": "failed to create container"})
                    return result
                else:
                    self.set_cycle(model_id=model_id, version=model_version)
                    async with self._lock:
                        self._deploy_states[model_key] = model_deploy_state
                        self._deploy_requests.remove((model_id, model_version))
                    print("success")
                    result = json.dumps({"code": 0, "msg": "success"})
                    return result
        # onload (read db) <-
        # logger

    async def add_container(self, model_id: str, version: str, container_num: int) -> json:
        model_key = model_id+":"+version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            print("the model is not deployed")
            result = json.dumps({"code": 8, "msg": "the model is not deployed"})
            return result

        async with self._lock:
            self._current_container_num += container_num

        deploy_count = 0
        for _ in range(container_num):
            http_port = self.get_port_http()
            grpc_port = self.get_port_grpc()
            container_name = model_id + "-" + uuid.uuid4().hex
            # loop = asyncio.get_event_loop()
            # container = await loop.run_in_executor(self._executor,
            #                                        functools.partial(self.run_container,
            #                                                          model_id=model_id,
            #                                                          container_name=container_name,
            #                                                          http_port=http_port,
            #                                                          grpc_port=grpc_port,
            #                                                          deploy_path=self._deploy_path + model_key + "/"))
            container = await self.run_container(model_id=model_id, container_name=container_name,
                                                 http_port=http_port, grpc_port=grpc_port,
                                                 deploy_path=self._deploy_path+model_key+"/")
            if container is None:
                break
            deploy_count += 1
            serving_container = ServingContainer(name=container_name, container=container,
                                                 http_url=(self._ip_container_server, http_port),
                                                 grpc_url=(self._ip_container_server, grpc_port))
            model_deploy_state.containers.append(serving_container)
        self.set_cycle(model_id=model_id, version=version)
        if deploy_count == container_num:
            print("add container success")
            result = json.dumps({"code": 200, "msg": "add container success"})
            return result
        else:
            async with self._lock:
                self._current_container_num -= (container_num - deploy_count)
            print("add container failed " + str(deploy_count) + "/" + str(container_num) + "is deployed")
            result = json.dumps({"code": 1, "msg": "add container failed " + str(deploy_count) + "/" + str(container_num) + "is deployed"})
            return result

    async def remove_container(self, model_id: str, version: str, container_num: int) -> json:
        model_key = model_id+":"+version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            print("the model is not deployed")
            result = json.dumps({"code": 8, "msg": "the model is not deployed"})
            return result

        containers = model_deploy_state.containers
        is_delete_success = True
        for _ in range(container_num):
            if len(containers) <= 1:
                result = await self.end_deploy(model_id=model_id, version=version)
                return result
            else:
                container = containers.pop()
                self.set_cycle(model_id=model_id, version=version)
                async with self._lock:
                    self._current_container_num -= container_num
                try:
                    container.container.stop()
                    container.container.remove()
                except APIError:
                    print("an error occur when remove container")
                    is_delete_success = False
        if is_delete_success:
            result = json.dumps({"code": 9, "msg": "an error occur when remove container"})
            return result
        else:
            result = json.dumps({"code": 200, "msg": "delete container success"})
            return result

    async def end_deploy(self, model_id: str, version: str) -> json:
        model_key = model_id+":"+version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            print("the model is not deployed")
            result = json.dumps({"code": 8, "msg": "the model is not deployed"})
            return result

        if model_deploy_state.ref_count <= 0:
            containers = model_deploy_state.containers
            for container in containers:
                container.container.remove(force=True)
            print("deploy ended: " + model_key)
            result = json.dumps({"code": 200, "msg": "deploy ended: " + model_key})
            async with self._lock:
                self._current_container_num -= len(containers)
            return result
        else:
            print("model is currently in use and cannot be closed")
            result = json.dumps({"code": 10, "msg": "model is currently in use and cannot be closed"})
            return result

    async def run_container(self, model_id: str, container_name: str, http_port: int, grpc_port: int, deploy_path: str):
        try:
            container = self._client.containers.run(image='tensorflow/serving:2.6.5', detach=True, name=container_name,
                                                    ports={'8501/tcp': http_port, '8500/tcp': grpc_port},
                                                    volumes=[deploy_path + model_id + ":/models/" + model_id],
                                                    environment=["MODEL_NAME=" + model_id]
                                                    )
            return container
        except ContainerError:
            print("the container exits with a non-zero exit code and detach is False")
            return None
        except ImageNotFound:
            print("the specified image does not exist")
            return None
        except APIError:
            print("the server returns an error")
            return None

    async def predict(self, model_id: str, model_version: str, data: json):
        model_deploy_state = self._deploy_states.get(model_id + ":" + model_version)
        if model_deploy_state is None:
            print("model is not in deploy state")
            return -1
        if model_deploy_state.cycle_iterator is None:
            print("cycle_iterator is not set")
            return -1
        url = next(model_deploy_state.cycle_iterator)
        async with Http() as http:
            async with self._lock:
                model_deploy_state.ref_count += 1
            result = await http.post(url, data)
            async with self._lock:
                model_deploy_state.ref_count -= 1
            return result

    def get_container_list(self):
        return self._client.containers.list()

    def get_container_names(self):
        containers = self.get_container_list()
        containers = [container.name for container in containers]
        return containers

    def remove_container_by_name(self, name: str):
        container = self._client.containers.get(name)
        container.remove(force=True)
        return 0

    async def get_model_state(self, model_name: str):
        url = self.get_model_endpoint(model_name)
        async with Http() as http:
            result = await http.get(url)
            result = result.decode('utf8').replace("'", '"')
            return result

    def init_client(self):
        self._ip_container_server = "localhost"  # config
        self._deploy_path = os.path.dirname(os.path.abspath(__file__)) + "/deploy/"  # config
        docker_host = os.getenv("DOCKER_HOST", default="unix:///run/user/1000/docker.sock")  # config
        self._client = docker.DockerClient(base_url=docker_host)  # delete
        self.on_load()  # delete
        self._client = docker.from_env()

    def on_load(self):
        for i in range(MAX_CONTAINER):
            self._grpc_port.append(GRPC_PORT_START + i)
            self._http_port.append(HTTP_PORT_START + i)

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
            if rpc_error.code() == grpc.StatusCode.CANCELLED:
                print("grpc cancelled")
            elif rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
                print("grpc unavailable")
            else:
                print("unknown error")
            return -1
        else:
            if response.status.error_code == 0:
                print("Reload sucessfully")
                return 0
            else:
                print("Reload failed!")
                print(response.status.error_code)
                print(response.status.error_message)
                return -1

    async def add_version(self, model_id: str, model_version: str):
        model_server = None
        for _ in self._deploy_states:
            if _.model_id == model_id:
                model_server = _
                break
        if model_server.state == StatusCode.ALREADY_EXIST:
            print("do add version progress: grpc")
            version_list = model_server.version
            for version in version_list:
                if version == model_version:
                    print("already deploy")
                    return 1
            print("add new version")
            result = self.copy_model(model_id, model_version)
            if result != 0:
                return -1
            model_server.versions.append(model_version)
            for container in model_server.containers:
                result = self.reset_version_config(host=container.grpc_url[0] + ":" + str(container.grpc_url[1]),
                                                   name=container.name,
                                                   base_path="/models/" + model_id, model_platform="tensorflow",
                                                   model_version_policy=model_server.versions)
                if result != 0:
                    print("reset config error : " + container.name)
            return 0
        else:
            print("model not deployed yet")
            return -1

    def copy_to_deploy(self, model_id: str, model_version: int) -> int:
        model_path = self._project_path + "/saved_models/" + model_id + "/" + model_version
        decoded_version = self.version_decode(model_version)
        deploy_key = model_id + ":" + decoded_version
        deploy_path = self._project_path + "/deploy/" + deploy_key + "/" + model_id + "/" + model_version
        try:
            copytree(model_path, deploy_path)
        except FileExistsError:
            print("version already exist in deploy dir")
            return -1
        except shutil.Error as err:
            src, dist, msg = err
            print("error on src:" + src + " target: " + dist + " | error: " + msg)
            return -1
        else:
            return 0

    def delete_model(self, model_id: str, model_version: str) -> int:
        deploy_path = self._project_path + "/deploy/" + model_id + "/" + model_version
        try:
            rmtree(deploy_path, ignore_errors=False)
        except shutil.Error as err:
            src, dist, msg = err
            print("error on src:" + src + " target: " + dist + " | error: " + msg)
            return -1
        else:
            return 0

    def set_cycle(self, model_id: str, version: str):
        urls = []
        model_key = model_id + ":" + version
        model_deploy_state = self._deploy_states.get(model_key)
        async with self._lock:
            for container in model_deploy_state.containers:
                url = "http://" + container.http_url[0] + ':' + str(container.http_url[1]) + \
                      "/v1/models/" + model_deploy_state.model_id + ":predict"
                urls.append(url)
            model_deploy_state.cycle_iterator = cycle(urls)

    def get_model_server(self, model_id: str, version: int):
        for model_server in self._deploy_states:
            if model_server.model_id == (model_id, version):
                return model_server
        return None

    def version_encode(self, version: str) -> int:
        sv = version.split('.')
        if len(sv[-1]) > 9:
            print("can't exceed decimal point over 9")
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


@dataclass
class ServingContainer:
    name: str
    container: Container
    http_url: tuple[str, int] = field(default_factory=tuple)
    grpc_url: tuple[str, int] = field(default_factory=tuple)


@dataclass
class ModelDeployState:
    model: tuple[str, int]
    state: int
    ref_count: int = 0
    cycle_iterator: object = None
    containers: list[ServingContainer] = field(default_factory=list)


class StatusCode(object):
    def __init__(self, *args, **kwargs):
        pass

    ALREADY_EXIST = 1
    IN_PROGRESS = 3
