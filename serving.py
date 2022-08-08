import json
import os
import asyncio
import shutil
import uuid
from dataclasses import dataclass, field
from shutil import copytree

import docker
import ray
import grpc
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.config import model_server_config_pb2
from asyncstdlib.itertools import cycle
# from aioshutil import copytree

from http_util import Http

HTTP_PORT_START = 8500
GRPC_PORT_START = 8000
MAX_CONTAINER = 20
DEPLOY_PATH = ""


@ray.remote
class ModelServing:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._deploy_requests = []
        self._deploy_states = []
        self._client = None
        self._http_port = []
        self._http_port_use = []
        self._grpc_port = []
        self._grpc_port_use = []
        self._ip_container_server = None
        self._deploy_path = None
        self.init_client()

    async def deploy(self, model_id: str, model_version: str, container_num: int) -> int:
        if container_num > MAX_CONTAINER:
            print("max container number exceeded")
            return 4

        async with self._lock:
            if model_id in self._deploy_requests:
                print("deploy duplicated")
                return 999
            else:
                self._deploy_requests.append(model_id)

        model_server = None
        for _ in self._deploy_states:
            if _.model_id == model_id:
                model_server = _
                break
        if model_server.state == StatusCode.ALREADY_EXIST:
            print("model already deployed. call add model interface")
        else:
            result = self.copy_model(model_id, model_version)
            if result != 0:
                return -1
            model_deploy_state = ModelDeployState(model_id=model_id, state=StatusCode.ALREADY_EXIST)
            model_deploy_state.versions.append(model_version)
            for _ in range(container_num):
                http_port = self.get_port_http()
                grpc_port = self.get_port_grpc()
                container_name = model_id + "-" + uuid.uuid4().hex
                container = await self.run_container(model_id=model_id, container_name=container_name,
                                                     http_port=http_port, grpc_port=grpc_port)
                serving_container = ServingContainer(name=container_name, container=container,
                                                     http_url=(self._ip_container_server, http_port),
                                                     grpc_url=(self._ip_container_server, grpc_port))
                model_deploy_state.containers.append(serving_container)
            self._deploy_states.append(model_deploy_state)

        self._deploy_requests.remove(model_id)
        return 0
        # get model_id is in deploy state -> block or not -> lock : blocking case : run same model container in same time
        # get model_id in deploy or not
        # if deploy add version -> check if version already serving -> remove all add container to container num
        # if not run container to container num
        # deploy state = [{model1:state}, {model2:state}]
        # one model_id can get multiple container model = [{cont1:state}, {cont2:state}]
        # one container can get multiple version cont = [v1, v2]
        # can stop certain version in container
        # when deploy, copy model file to deploy folder -> run or grpc
        # when remove, delete versions, if version less then one remove container
        # cp
        # add folder to deploy(if not exist
        # async cycle package <-
        # predict
        # onload

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
            for _ in model_server.containers:
                result = self.reset_version_config(host=self._ip_container_server, name=_.name,
                                                   base_path="/models/" + model_id, model_platform="tensorflow",
                                                   model_version_policy=model_server.versions)
                if result != 0:
                    print("reset config error : " + _.name)
                    return 4 # copy ok
            return 0
        else:
            print("model not deployed yet")
            return -1

    async def run_container(self, model_id: str, container_name: str, http_port: int, grpc_port: int):
        container = self._client.containers.run(image='tensorflow/serving:2.6.5', detach=True, name=container_name,
                                                ports={'8501/tcp': http_port, '8500/tcp': grpc_port},
                                                volumes=[self._deploy_path + model_id + ":/models/" + model_id],
                                                environment=["MODEL_NAME=" + model_id]
                                                )
        return container

    async def predict(self, model_id: str, model_version: str, data: json):

        return 0

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

    def get_model_endpoint(self, model_name: str):
        return "http://" + self._ip_container_server + ":8501/v1/models/" + model_name  # delete

    def init_client(self):
        self._ip_container_server = "localhost"  # config
        self._deploy_path = os.path.dirname(os.path.abspath(__file__)) + "/saved_models/"  # config
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
        # host = "localhost:8500"
        # name = "test"
        # base_path = "/models/test"
        # model_platform = "tensorflow"
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

    def copy_model(self, model_id: str, model_version: str) -> int:
        try:
            project_path = os.path.dirname(os.path.abspath(__file__))
            model_path = project_path + "/saved_models/" + model_id + "/" + model_version
            deploy_path = project_path + "/deploy/" + model_id + "/" + model_version
            copytree(model_path, deploy_path)
        except FileExistsError:
            print("version already exist in deploy dir")
            return 4
        except shutil.Error as err:
            src, dist, msg = err
            print("error on src:" + src + " target: " + dist + " | error: " + err)
            return -1
        else:
            return 0

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
    container: object
    http_url: tuple[str, int] = field(default_factory=tuple)
    grpc_url: tuple[str, int] = field(default_factory=tuple)


@dataclass
class ModelDeployState:
    model_id: str
    state: int
    versions: list[str] = field(default_factory=list)
    containers: list[ServingContainer] = field(default_factory=list)


class StatusCode(object):
    def __init__(self, *args, **kwargs):
        pass

    ALREADY_EXIST = 1
    IN_PROGRESS = 3
