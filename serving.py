import os
import asyncio

import docker
import ray

from http_util import Http

HTTP_PORT_START = 8500
GRPC_PORT_START = 8000
MAX_CONTAINER = 20
DEPLOY_PATH = ""


@ray.remote
class ModelServing:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._deploy_state = None
        self._client = None
        self._http_port = []
        self._http_port_use = []
        self._grpc_port = []
        self._grpc_port_use = []
        self.init_client()


    def deploy(self, model_id:str, model_version:str, container_num:int) -> int:
        async with self._lock:
            print("some")
        #get model_id is in deploy state -> block or not -> lock
        #get model_id in deploy or not
        #if deploy add version -> check if version already serving -> remove all add container to container num
        #if not run container to container num
        #deploy state = [{model1:state}, {model2:state}]
        #one model_id can get multiple container model = [{cont1:state}, {cont2:state}]
        #one container can get multiple version cont = [v1, v2]
        #can stop certain version in container
        #when deploy, copy model file to deploy folder -> run or grpc
        #when remove, delete versions, if version less then one remove container
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

    def get_port_http(self, container: ):
        self._http_port_use.append(self._http_port.pop())

    def get_port_grpc(self, container: str):
        self._grpc_port_use.append(self._grpc_port.pop())

    def release_port_http(self):
        self._http_port
