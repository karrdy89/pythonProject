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
from apscheduler.schedulers.asyncio import AsyncIOScheduler

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
        self._deploy_requests: list[tuple[str, str]] = []   # tuple of model_id and version
        self._deploy_states: dict[str, ModelDeployState] = {}
        self._gc_list: list[tuple[int, str]] = []   # tuple of manage type and key
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
        self.init_client()

    async def deploy(self, model_id: str, version: str, container_num: int) -> json:
        print("deploy start")
        if (model_id, version) in self._deploy_requests:
            print("same deploy request is in progress")
            print(datetime.datetime.now())
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
                    print("nothing to change")
                    result = json.dumps({"code": 4, "msg": "nothing to change",
                                         "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                        'utf8')
                    return result

        if container_num <= 0:
            self._deploy_requests.remove((model_id, version))
            print("nothing to change")
            result = json.dumps({"code": 4, "msg": "nothing to change",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                'utf8')
            return result
        if (self._current_container_num + container_num) > MAX_CONTAINER:
            print("max container number exceeded")
            result = json.dumps({"code": 4, "msg": "max container number exceeded",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                'utf8')
            return result

        cp_result = self.copy_to_deploy(model_id, encoded_version)
        if cp_result == -1:
            print("an error occur when copying model file")
            result = json.dumps({"code": 7, "msg": "n error occur when copying model file",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                'utf8')
            async with self._lock:
                self._current_container_num -= container_num
            return result

        model_deploy_state = ModelDeployState(model=(model_id, encoded_version), state=StateCode.AVAILABLE)
        result = await self.deploy_containers(model_id, version, container_num, model_deploy_state)
        return result

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
        if (self._current_container_num + container_num) > MAX_CONTAINER:
            print("max container number exceeded")
            result = json.dumps({"code": 4, "msg": "max container number exceeded",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode(
                'utf8')
            return result

        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            print("the model is not deployed")
            result = json.dumps({"code": 8, "msg": "the model is not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        result = await self.deploy_containers(model_id, version, container_num, model_deploy_state)
        return result

    async def remove_container(self, model_id: str, version: str, container_num: int) -> json:
        # make is model exist api
        # change to use just non use flag, add del list
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            print("the model is not deployed")
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
            print("shutdown pending on " + str(container_num) + " containers")
            result = json.dumps({"code": 200, "msg": "shutdown pending on " + str(container_num) + " containers",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

    async def end_deploy(self, model_id: str, version: str) -> json:
        encoded_version = self.version_encode(version)
        model_key = model_id + "_" + version
        model_deploy_state = self._deploy_states.get(model_key)
        if model_deploy_state is None:
            print("the model is not deployed")
            result = json.dumps({"code": 8, "msg": "the model is not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        model_deploy_state.state = StateCode.SHUTDOWN
        self._gc_list.append((ManageType.MODEL, model_key))
        self.delete_model(model_id, encoded_version)
        print("end_deploy accepted")
        result = json.dumps({"code": 200, "msg": "end_deploy accepted",
                             "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
        return result

    async def predict(self, model_id: str, version: str, data: dict):
        model_deploy_state = self._deploy_states.get(model_id + "_" + version)
        if (model_deploy_state is None) or (model_deploy_state.cycle_iterator is None):
            print("the model not deployed")
            result = json.dumps({"code": 10, "msg": "the model not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        if model_deploy_state.state == StateCode.SHUTDOWN:
            print("the model not deployed")
            result = json.dumps({"code": 10, "msg": "the model not deployed",
                                 "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')
            return result

        deploy_info = next(model_deploy_state.cycle_iterator)
        url = deploy_info[0]
        state = deploy_info[1]
        container_name = deploy_info[2]
        print(url)
        container = model_deploy_state.containers[container_name]
        if state == StateCode.AVAILABLE:
            async with Http() as http:
                async with self._lock:
                    container.ref_count += 1
                result = await http.post_json(url, data)
                if result is None:
                    print("http request error")
                    result = json.dumps({"code": 10, "msg": "http request error"}).encode('utf8')
                elif result == -1:
                    print("connection error trying to fail back")
                    result = json.dumps({"code": 11, "msg": "connection error trying to fail back"}).encode('utf8')
                    await self.fail_back(model_id, version, container_name)
                    result = await self.predict(model_id, version, data)
                async with self._lock:
                    container.ref_count -= 1
                print(result)
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
                print(list_http_url[i], list_container[i])
                serving_container = ServingContainer(name=list_container_name[i], container=list_container[i],
                                                     http_url=list_http_url[i], grpc_url=list_grpc_url[i],
                                                     state=StateCode.AVAILABLE)
                model_deploy_state.containers[list_container_name[i]] = serving_container
                deploy_count += 1
        print(model_deploy_state.containers)

        async with self._lock:
            self._current_container_num -= (container_num - deploy_count)
        self._deploy_states[model_key] = model_deploy_state
        self._deploy_requests.remove((model_id, version))
        await self._set_cycle(model_id=model_id, version=version)
        print("deploy finished. " + str(deploy_count) + "/" + str(container_num) + " is deployed")
        print(datetime.datetime.now())
        result = json.dumps({"code": 0,
                             "msg": "deploy finished. " + str(deploy_count) + "/" + str(container_num) + "deployed",
                             "event_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode('utf8')

        return result

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

    async def fail_back(self, model_id: str, version: str, container_name: str):
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

    def init_client(self):
        self._ip_container_server = "localhost"  # config
        self._deploy_path = os.path.dirname(os.path.abspath(__file__)) + "/deploy/"  # config
        # docker_host = os.getenv("DOCKER_HOST", default="unix:///run/user/1000/docker.sock")  # config
        # self._client = docker.DockerClient(base_url=docker_host)  # delete
        self.on_load()  # delete
        self._client = docker.from_env()
        self._manager_handle = AsyncIOScheduler()
        self._manager_handle.add_job(self.gc_container, "interval", seconds=CHECK_INTERVAL, id="gc_container")
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
            if rpc_error.code() == grpc.StateCode.CANCELLED:
                print("grpc cancelled")
            elif rpc_error.code() == grpc.StateCode.UNAVAILABLE:
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
        if model_server.state == StateCode.ALREADY_EXIST:
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
        decoded_version = self.version_decode(version)
        deploy_key = model_id + "_" + decoded_version
        deploy_path = self._project_path + "/deploy/" + deploy_key + "/" + model_id + "/" + str(version)
        try:
            rmtree(deploy_path, ignore_errors=False)
        except shutil.Error as err:
            src, dist, msg = err
            print("error on src:" + src + " target: " + dist + " | error: " + msg)
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
            print(cycle_list)
            model_deploy_state.cycle_iterator = cycle(cycle_list)
        return 0

    async def gc_container(self):
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
                        except APIError:
                            print("an error occur when remove container")
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
                    print("model deploy end")
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
                                except APIError:
                                    print("an error occur when remove container")
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
                                print("container deleted")
                        else:
                            await self.end_deploy(model_id, version)
                            self._gc_list.remove(type_key)
                    else:
                        continue
        print(self._deploy_states)

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
    state: int
    ref_count: int = 0
    http_url: tuple[str, int] = field(default_factory=tuple)
    grpc_url: tuple[str, int] = field(default_factory=tuple)


@dataclass
class ModelDeployState:
    model: tuple[str, int]
    state: int
    # ref_count: int = 0
    cycle_iterator = None
    containers: dict = field(default_factory=dict)
    # containers: list[ServingContainer] = field(default_factory=list)


class StateCode(object):
    def __init__(self, *args, **kwargs):
        pass

    ALREADY_EXIST = 1
    IN_PROGRESS = 3
    AVAILABLE = 0
    SHUTDOWN = 4


class ManageType:
    def __init__(self, *args, **kwargs):
        pass

    MODEL = 0
    CONTAINER = 1
