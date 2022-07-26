import os
import asyncio

import docker

from HttpService import Http


class ModelServing:
    def __init__(self):
        self._deploy_state = None
        self._client = None
        self.init_client()

    def run_container(self, name: str):
        # change path when deploy
        saved_model_path = os.path.dirname(os.path.abspath(__file__)) + "/saved_models/"
        self._client.containers.run(image='tensorflow/serving:2.6.5', detach=True, name=name,
                                    ports={'8501/tcp': '8501/tcp'},
                                    volumes=[saved_model_path + name + ":/models/" + name],
                                    environment=["MODEL_NAME="+name]
                                    )
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
            await asyncio.sleep(5)
            result = await http.get(url)
            result = result.decode('utf8').replace("'", '"')
            return result

    @staticmethod
    def get_model_endpoint(model_name: str):
        return "http://localhost:8501/v1/models/"+model_name

    def init_client(self):
        # change path when deploy
        docker_host = os.getenv("DOCKER_HOST", default="unix:///run/user/1000/docker.sock")
        self._client = docker.DockerClient(base_url=docker_host)
        # client = docker.from_env()
