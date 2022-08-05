import json
import os
import asyncio
from dataclasses import dataclass
from shutil import copytree

import docker
import ray
import grpc
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.config import model_server_config_pb2

from http_util import Http

HTTP_PORT_START = 8500
GRPC_PORT_START = 8000
MAX_CONTAINER = 20
DEPLOY_PATH = ""


@ray.remote
class ModelServing:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._deploy_state = []
        self._client = None
        self._http_port = []
        self._http_port_use = []
        self._grpc_port = []
        self._grpc_port_use = []
        self.init_client()


    async def deploy(self, model_id:str, model_version:str, container_num:int) -> int:
        is_deploy = False
        is_progress = False
        async with self._lock:
            for model_server in self._deploy_state:
                if model_server.model_id == model_id:
                    is_deploy = True
                    is_progress = (ModelDeployState.state == "progress")
                    break
        if is_progress:
            return 999 #"same model is in deploy progress"

        if is_deploy:
            print("do add version progress: grpc")
        else:
            print("do run container progress: docker")

        #get model_id is in deploy state -> block or not -> lock : blocking case : run same model container in same time
        #get model_id in deploy or not
        #if deploy add version -> check if version already serving -> remove all add container to container num
        #if not run container to container num
        #deploy state = [{model1:state}, {model2:state}]
        #one model_id can get multiple container model = [{cont1:state}, {cont2:state}]
        #one container can get multiple version cont = [v1, v2]
        #can stop certain version in container
        #when deploy, copy model file to deploy folder -> run or grpc
        #when remove, delete versions, if version less then one remove container
        #async cycle package
        self.run_container(model_id=model_id, container_name=model_id, )
        return 0

    def run_container(self, model_id: str, container_name: str, port_http: int, port_grpc: int) -> int:
        saved_model_path = os.path.dirname(os.path.abspath(__file__)) + "/saved_models/"
        container = self._client.containers.run(image='tensorflow/serving:2.6.5', detach=True, name=container_name,
                                                ports={'8501/tcp': port_http, '8500/tcp': port_grpc},
                                                volumes=[saved_model_path + model_id + ":/models/" + model_id],
                                                environment=["MODEL_NAME=" + model_id]
                                                )
        return container

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

    @staticmethod
    def get_model_endpoint(model_name: str):
        return "http://localhost:8501/v1/models/" + model_name

    def init_client(self):
        docker_host = os.getenv("DOCKER_HOST", default="unix:///run/user/1000/docker.sock")
        self._client = docker.DockerClient(base_url=docker_host)
        self._client = docker.from_env()

    def on_load(self):
        for i in range(MAX_CONTAINER):
            self._grpc_port.append(GRPC_PORT_START + i)
            self._http_port.append(HTTP_PORT_START + i)

    def get_port_http(self, container: str):
        self._http_port_use.append(self._http_port.pop())

    def get_port_grpc(self, container: str):
        self._grpc_port_use.append(self._grpc_port.pop())

    def release_port_http(self):
        self._http_port

    # def reset_version_config(self, host: str, name: str, base_path: str,
    #                          model_platform: str, model_version_policy: dict):
    def reset_version_config(self):
        host = "localhost:8500"
        name = "test"
        base_path = "/models/test"
        model_platform = "tensorflow"
        model_versions = 1
        channel = grpc.insecure_channel(host)
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        request = model_management_pb2.ReloadConfigRequest()
        model_server_config = model_server_config_pb2.ModelServerConfig()

        config_list = model_server_config_pb2.ModelConfigList()
        config = config_list.config.add()
        config.name = name
        config.base_path = base_path
        config.model_platform = model_platform
        config.model_version_policy.specific.versions.append(1)

        model_server_config.model_config_list.CopyFrom(config_list)
        request.config.CopyFrom(model_server_config)
        try:
            response = stub.HandleReloadConfigRequest(request, 20)
        except response.status.error_message:
            print(response.status.error_message)
        if response.status.error_code == 0:
            print("Reload sucessfully")
        else:
            print("Reload failed!")
            print(response.status.error_code)
            print(response.status.error_message)
        return 0


@dataclass
class ModelDeployState:
    model_id: str
    state: str
    containers: list
    versions: list

