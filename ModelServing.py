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
        self._client.containers.run(image='tensorflow/serving:2.6.5', detach=True, name=name,
                              ports={'8501/tcp': '8501/tcp'}, volumes=["/home/ky/PycharmProjects/pythonProject/saved_models/test:/models/test"],
                                    environment=["MODEL_NAME=test"])
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
            print(result)
            return result

    def get_model_endpoint(self, model_name: str):
        return "http://localhost:8501/v1/models/test/"

    def init_client(self):
        docker_host = os.getenv("DOCKER_HOST", default="unix:///run/user/1000/docker.sock")
        self._client = docker.DockerClient(base_url=docker_host)
        # client = docker.from_env()