import json
import os
import asyncio
import shutil
import uuid
import functools
import datetime
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
from docker.errors import ContainerError, ImageNotFound, APIError
from docker.models.containers import Container
from apscheduler.schedulers.background import BackgroundScheduler

from http_util import Http

HTTP_PORT_START = 8500
GRPC_PORT_START = 8000
MAX_CONTAINER = 20
CHECK_INTERVAL = 10
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
        self._manager_handle = None
        self.init_client()

    async def deploy(self, model_id: str, version: str, container_num: int) -> json:
        print("deploy start")
        await asyncio.sleep(5)
        if (self._current_container_num + container_num) > MAX_CONTAINER:
            print("max container number exceeded")
            result = json.dumps({"code": 4, "msg": "max container number exceeded",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        if (model_id, version) in self._deploy_requests:
            print("same deploy request is in progress")
            print(datetime.datetime.now())
            result = json.dumps({"code": 5, "msg": "same deploy request is in progress",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            return result
        else:
            self._deploy_requests.append((model_id, version))
            encoded_version = self.version_encode(version)
            model_key = model_id + "_" + version
            model_deploy_state = self._deploy_states.get(model_key)
            if model_deploy_state is not None:
                if model_deploy_state.state == StatusCode.ALREADY_EXIST:
                    print(datetime.datetime.now())
                    print("model already deployed.")
                    result = json.dumps({"code": 6, "msg": "model already deployed",
                                         "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                        'utf8')
                    return result

            async with self._lock:
                self._current_container_num += container_num
            result = self.copy_to_deploy(model_id, encoded_version)
            if result == -1:
                print("an error occur when copying model file")
                result = json.dumps({"code": 7, "msg": "n error occur when copying model file",
                                     "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                    'utf8')
                async with self._lock:
                    self._current_container_num -= container_num
                return result
            model_deploy_state = ModelDeployState(model=(model_id, encoded_version), state=StatusCode.ALREADY_EXIST)

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
                # container = await self.run_container(model_id=model_id, container_name=container_name,
                #                                      http_port=http_port, grpc_port=grpc_port,
                #                                      deploy_path=self._deploy_path+model_key+"/")
                # if container is None:
                #     is_deploy_failed = True
                #     break
                # serving_container = ServingContainer(name=container_name, container=container,
                #                                      http_url=(self._ip_container_server, http_port),
                #                                      grpc_url=(self._ip_container_server, grpc_port))
                # model_deploy_state.containers.append(serving_container)
            list_container = await asyncio.gather(*futures)
            deploy_count = 0
            for i in range(len(list_container)):
                if list_container[i] is not None:
                    print(list_http_url[i], list_container[i])
                    serving_container = ServingContainer(name=list_container_name[i], container=list_container[i],
                                                         http_url=list_http_url[i], grpc_url=list_grpc_url[i],
                                                         state="available")
                    model_deploy_state.containers[list_container_name[i]] = serving_container
                    # model_deploy_state.containers.append(serving_container)
                    deploy_count += 1
            print(model_deploy_state.containers)
            async with self._lock:
                self._current_container_num -= (container_num - deploy_count)
            self._deploy_states[model_key] = model_deploy_state
            self._deploy_requests.remove((model_id, version))
            await self.set_cycle(model_id=model_id, version=version)
            print("deploy finished. " + str(deploy_count) + "/" + str(container_num) + " is deployed")
            print(datetime.datetime.now())
            result = json.dumps({"code": 0,
                                 "msg": "deploy finished. " + str(deploy_count) + "/" + str(container_num) + "deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

            # if is_deploy_failed:
            #     for serving_container in model_deploy_state.containers:
            #         serving_container.container.remove(force=True)
            #     # async with self._lock:
            #     #     self._current_container_num -= container_num
            #     print("failed to create container")
            #     result = json.dumps({"code": 8, "msg": "failed to create container",
            #                          "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
            #         'utf8')
            #     return result
            # else:
            #     self._deploy_states[model_key] = model_deploy_state
            #     self._deploy_requests.remove((model_id, version))
            #     await self.set_cycle(model_id=model_id, version=version)
            #     print("success")
            #     result = json.dumps({"code": 0, "msg": "success",
            #                          "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
            #         'utf8')
            #     return result

        # kill pending(set container state to kill and add ref count to container) -> kill in GC thread(run thread when init class) <-
        #   > container : state, ref_count, url, port, key
        #   > can get container state in predict method : cycle with pair(url, name? = use this time or idx??=possible, or state?(y) - cycle, add, delete, deploy)? mds(model,version,cycle,containerlist)
        #   > start and stop api save thread handle
        #   > can change container state in predict method -> less update in better so find mds's container. predict know mds already so, find container in container list
        #   > model_deploy_state : state, ref_count(sum of container ref_count)
        #   > GC Target : all containers - where to be placed
        #   > GC job : check state, delete, update <- check gc in remote working
        # errors in fail back process
        # debug(can remove safely, is cycle work fine)
        # onload (read db)
        # logger

    async def get_deploy_state(self) -> json:
        # add conatiner and state to result
        print("get deploy state")
        deploy_states = []
        for key in self._deploy_states:
            sep_key = key.split("_")
            model_id = sep_key[0]
            version = sep_key[1]
            model_deploy_state = self._deploy_states[key]
            container_num = len(model_deploy_state.containers)
            deploy_state = {"model_id": model_id, "version": version, "container_num": container_num}
            deploy_states.append(deploy_state)
        await asyncio.sleep(3)
        return json.dumps({"deploy_states": deploy_states,
                           "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')

    async def add_container(self, model_id: str, version: str, container_num: int) -> json:
        #check if exceed, intergrate with deploy
        #make is model exist api
        #change for to gather
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            print("the model is not deployed")
            result = json.dumps({"code": 8, "msg": "the model is not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        async with self._lock:
            self._current_container_num += container_num

        deploy_count = 0
        for _ in range(container_num):
            http_port = self.get_port_http()
            grpc_port = self.get_port_grpc()
            container_name = model_id + "-" + uuid.uuid4().hex
            container = self.run_container(model_id=model_id, container_name=container_name,
                                           http_port=http_port, grpc_port=grpc_port,
                                           deploy_path=self._deploy_path + model_key + "/")
            if container is None:
                break
            deploy_count += 1
            serving_container = ServingContainer(name=container_name, container=container,
                                                 http_url=(self._ip_container_server, http_port),
                                                 grpc_url=(self._ip_container_server, grpc_port))
            model_deploy_state.containers.append(serving_container)
        await self.set_cycle(model_id=model_id, version=version)
        if deploy_count == container_num:
            print("add container success")
            result = json.dumps({"code": 200, "msg": "add container success",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result
        else:
            async with self._lock:
                self._current_container_num -= (container_num - deploy_count)
            print("add container failed " + str(deploy_count) + "/" + str(container_num) + "is deployed")
            result = json.dumps({"code": 1, "msg": "add container failed " + str(deploy_count) + "/" + str(
                container_num) + "is deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

    async def remove_container(self, model_id: str, version: str, container_num: int) -> json:
        #make is model exist api
        #change to use just non use flag
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            print("the model is not deployed")
            result = json.dumps({"code": 8, "msg": "the model is not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        containers = model_deploy_state.containers
        is_delete_success = True
        for _ in range(container_num):
            if len(containers) <= 1:
                result = await self.end_deploy(model_id=model_id, version=version)
                return result
            else:
                container = containers.pop()
                await self.set_cycle(model_id=model_id, version=version)
                async with self._lock:
                    self._current_container_num -= container_num
                http_port = container.http_url[1]
                grpc_port = container.grpc_url[1]
                self.release_port_http(http_port)
                self.release_port_grpc(grpc_port)
                try:
                    container.container.stop()
                    container.container.remove()
                except APIError:
                    print("an error occur when remove container")
                    is_delete_success = False
        if is_delete_success:
            result = json.dumps({"code": 9, "msg": "an error occur when remove container",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result
        else:
            result = json.dumps({"code": 200, "msg": "delete container success",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

    async def end_deploy(self, model_id: str, version: str) -> json:
        #make is model exist api
        #change to use just non use flag
        encoded_version = self.version_encode(version)
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            print("the model is not deployed")
            result = json.dumps({"code": 8, "msg": "the model is not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        if model_deploy_state.ref_count <= 0:
            containers = model_deploy_state.containers
            for container in containers:
                self.release_port_grpc(container.grpc_url[1])
                self.release_port_http(container.http_url[1])
                container.container.remove(force=True)
            print("deploy ended: " + model_key)
            result = json.dumps({"code": 200, "msg": "deploy ended: " + model_key,
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            self.delete_model(model_id, encoded_version)
            async with self._lock:
                del self._deploy_states[model_key]
                self._current_container_num -= len(containers)
            return result
        else:
            print("model is currently in use and cannot be closed")
            result = json.dumps({"code": 10, "msg": "model is currently in use and cannot be closed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

    async def predict(self, model_id: str, version: str, data: dict):
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if (model_deploy_state is None) or (model_deploy_state.cycle_iterator is None):
            print("the model not deployed")
            result = json.dumps({"code": 10, "msg": "the model not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        deploy_info = next(model_deploy_state.cycle_iterator)
        url = deploy_info[0]
        state = deploy_info[1]
        container_name = deploy_info[2]
        container = model_deploy_state.containers[container_name]
        if state == "available":
            async with Http() as http:
                async with self._lock:
                    container.ref_count += 1
                result = await http.post_json(url, data)
                print(result)
                if result is None:
                    print("http request error")
                    result = json.dumps({"result": []}).encode('utf8')
                    print("trying to fail back")    #on network error
                    await self.fail_back(model_id, version, url)
                async with self._lock:
                    model_deploy_state.ref_count -= 1
                return result  # if need to change from byte to json

    def run_container(self, model_id: str, container_name: str, http_port: int, grpc_port: int, deploy_path: str):
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
        except APIError as error:
            print("the server returns an error")
            print(error.__str__())
            return None

    async def fail_back(self, model_id: str, version: str, url: str):
        url_sep = url.split(":")
        url_port = url_sep[2].split("/")[0]
        url_port = int(url_port)
        for key in self._deploy_states:
            model_deploy_state = self._deploy_states[key]
            for container in model_deploy_state.containers:
                container_port = container.http_url[1]
                if url_port == container_port:
                    model_deploy_state.containers.remove(container)
                    break
        await self.set_cycle(model_id, version)
        await self.add_container(model_id, version, 1)

    async def get_model_state(self, model_name: str):
        url = self.get_model_endpoint(model_name)
        async with Http() as http:
            result = await http.get(url)
            result = result.decode('utf8').replace("'", '"')
            return result

    def init_client(self):
        self._ip_container_server = "localhost"  # config
        self._deploy_path = os.path.dirname(os.path.abspath(__file__)) + "/deploy/"  # config
        # docker_host = os.getenv("DOCKER_HOST", default="unix:///run/user/1000/docker.sock")  # config
        # self._client = docker.DockerClient(base_url=docker_host)  # delete
        self.on_load()  # delete
        self._client = docker.from_env()
        self._manager_handle = BackgroundScheduler()
        self._manager_handle.add_job(self.container_manage, "interval", seconds=CHECK_INTERVAL, id="container_manager")
        self._manager_handle.start()

    def on_load(self):
        for i in range(MAX_CONTAINER * 2):
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

    async def add_version(self, model_id: str, version: str):
        model_server = None
        for _ in self._deploy_states:
            if _.model_id == model_id:
                model_server = _
                break
        if model_server.state == StatusCode.ALREADY_EXIST:
            print("do add version progress: grpc")
            version_list = model_server.version
            for v in version_list:
                if v == version:
                    print("already deploy")
                    return 1
            print("add new version")
            result = self.copy_to_deploy(model_id, version)
            if result != 0:
                return -1
            model_server.versions.append(version)
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

    def get_deploy_state(self, model_id: str, version: str):
        return self._deploy_states[model_id + ":" + version]

    def copy_to_deploy(self, model_id: str, version: int) -> int:
        model_path = self._project_path + "/saved_models/" + model_id + "/" + str(version)
        decoded_version = self.version_decode(version)
        deploy_key = model_id + "_" + decoded_version
        deploy_path = self._project_path + "/deploy/" + deploy_key + "/" + model_id + "/" + str(version)
        try:
            copytree(model_path, deploy_path)
        except FileExistsError:
            print("version already exist in deploy dir")
            return 1
        except shutil.Error as err:
            src, dist, msg = err
            print("error on src:" + src + " target: " + dist + " | error: " + msg)
            return -1
        else:
            return 0

    def delete_model(self, model_id: str, version: int) -> int:
        deploy_path = self._project_path + "/deploy/" + model_id + "/" + str(version)
        try:
            rmtree(deploy_path, ignore_errors=False)
        except shutil.Error as err:
            src, dist, msg = err
            print("error on src:" + src + " target: " + dist + " | error: " + msg)
            return -1
        else:
            return 0

    async def set_cycle(self, model_id: str, version: str):
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
            print(cycle_list)
            model_deploy_state.cycle_iterator = cycle(cycle_list)

    def container_manage(self):
        print(self._deploy_states)

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
    state: str
    ref_count: int = 0
    http_url: tuple[str, int] = field(default_factory=tuple)
    grpc_url: tuple[str, int] = field(default_factory=tuple)


@dataclass
class ModelDeployState:
    model: tuple[str, int]
    state: int
    # ref_count: int = 0
    cycle_iterator: object = None
    containers: dict = field(default_factory=dict)
    # containers: list[ServingContainer] = field(default_factory=list)


class StatusCode(object):
    def __init__(self, *args, **kwargs):
        pass

    ALREADY_EXIST = 1
    IN_PROGRESS = 3
