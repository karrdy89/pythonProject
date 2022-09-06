import json
import os

import ray
import httpx
from fastapi import FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

import VO.request_vo as rvo
from pipeline import Pipeline
from tensorboard_service import TensorBoardTool
from utils.common import version_decode, version_encode
from pipeline import TrainInfo

project_path = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(CORSMiddleware)
router = InferringRouter()


@cbv(router)
class AIbeemRouter:
    """
    A class that runs and manages the tensorboard service.

    Attributes
    ----------
    _worker : str
        Class name of instance.
    _server : actor
        An actor handle of model_serving.
    _logger : actor
        An actor handle of global logger.
    _shared_state : actor
        An actor handle of global data store.
    _tensorboard_tool : TensorBoardTool
        An instance of tensorboard service

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    @router:
        API endpoints of server
    """
    def __init__(self):
        self._worker = type(self).__name__
        self._server: ray.actor = ray.get_actor("model_serving")
        self._logger: ray.actor = ray.get_actor("logging_service")
        self._shared_state: ray.actor = ray.get_actor("shared_state")
        self._tensorboard_tool: TensorBoardTool = TensorBoardTool()

    @router.post("/train/run")
    async def train(self, request_body: rvo.Train):
        model = request_body.model_id
        version = request_body.version
        pipeline_name = model + ":" + version
        if ray.get(self._shared_state.is_actor_exist.remote(name=pipeline_name)):
            return "same model is training"
        train_info = TrainInfo()
        train_info.name = pipeline_name
        train_info.epoch = request_body.epoch
        train_info.early_stop = request_body.early_stop
        train_info.data_split = request_body.data_split
        train_info.batch_size = request_body.batch_size
        tmp_path = model + "/" + str(version_encode(version))
        train_info.save_path = project_path + '/saved_models/' + tmp_path
        train_info.log_path = project_path + '/train_logs/' + tmp_path
        pipeline_actor = Pipeline.options(name="pipeline_name").remote()
        pipeline_actor.run_pipeline.remote(name=model, version=version, train_info=train_info)
        self._shared_state.set_actor.remote(name=pipeline_name, act=pipeline_actor)
        return "train started"

    @router.post("/train/stop")
    async def stop_train(self, request_body: rvo.BasicModelInfo):
        model = request_body.model_id
        version = request_body.version
        pipeline_name = model + ":" + version
        result = await self._shared_state.kill_actor.remote(name=pipeline_name)
        if result == 0:
            return "success"
        else:
            return "fail"

    @router.get("/train/state")
    async def get_train_state(self, request_body: rvo.BasicModelInfo):
        model = request_body.model_id
        version = request_body.version
        pipeline_name = model + ":" + version
        pipeline_state = await self._shared_state.get_pipeline_result.remote(name=pipeline_name)
        train_result = await self._shared_state.get_train_result.remote(name=pipeline_name)
        result = {"pipeline_state": pipeline_state, "train_result": train_result}
        result = json.dumps(result)
        return result

    @router.get("/dataset/make")
    async def get_train_info(self):
        pass

    @router.get("/dataset/download")
    async def get_train_info(self):
        pass

    @router.post("/deploy")
    async def deploy(self, request_body: rvo.Deploy):
        print("accept deploy request")
        remote_job_obj = self._server.deploy.remote(model_id=request_body.model_id,
                                                    version=request_body.version,
                                                    container_num=request_body.container_num)
        result = await remote_job_obj
        return result

    @router.get("/deploy/state")
    async def get_deploy_state(self):
        remote_job_obj = self._server.get_deploy_state.remote()
        result = await remote_job_obj
        return result

    @router.post("/deploy/add_container")
    async def add_container(self, request_body: rvo.AddContainer):
        remote_job_obj = self._server.add_container.remote(model_id=request_body.model_id, version=request_body.version,
                                                           container_num=request_body.container_num)
        result = await remote_job_obj
        return result

    @router.post("/deploy/remove_container")
    async def remove_container(self, request_body: rvo.RemoveContainer):
        remote_job_obj = self._server.remove_container.remote(model_id=request_body.model_id,
                                                              version=request_body.version,
                                                              container_num=request_body.container_num)
        result = await remote_job_obj
        return result

    @router.post("/deploy/end_deploy")
    async def end_deploy(self, request_body: rvo.EndDeploy):
        remote_job_obj = self._server.end_deploy.remote(model_id=request_body.model_id, version=request_body.version)
        result = await remote_job_obj
        return result

    @router.post("/predict")
    async def predict(self, request_body: rvo.Predict):
        remote_job_obj = self._server.predict.remote(model_id=request_body.model_id, version=request_body.version,
                                                     data=request_body.feature)
        result = await remote_job_obj
        return result

    @router.post("/tensorboard")
    async def create_tensorboard(self, request_body: rvo.BasicModelInfo):
        version = request_body.version
        encoded_version = version_encode(version)
        model = request_body.model_id
        log_path = project_path + "/train_logs/" + model + "/" + str(encoded_version)
        if os.path.isdir(log_path):
            port = self._tensorboard_tool.run(dir_path=log_path)
            path = "/tensorboard/" + str(port)
        else:
            path = "log file not exist"
        return path


async def _reverse_proxy(request: Request):
    path = request.url.path.split('/')
    port = path[2]
    path = '/'.join(path[3:])
    client = httpx.AsyncClient(base_url="http://127.0.0.1:" + port)
    url = httpx.URL(path=path,
                    query=request.url.query.encode("utf-8"))
    rp_req = client.build_request(request.method, url,
                                  headers=request.headers.raw,
                                  content=await request.body())
    try:
        rp_resp = await client.send(rp_req, stream=True)
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}.")
        return f"An error occurred while requesting {exc.request.url!r}."
    else:
        return StreamingResponse(
            rp_resp.aiter_raw(),
            status_code=rp_resp.status_code,
            headers=rp_resp.headers,
            background=BackgroundTask(rp_resp.aclose),
        )


app.add_route("/dashboard/{port}/{path:path}", _reverse_proxy, ["GET", "POST"])
app.add_route("/tensorboard/{port}/{path:path}", _reverse_proxy, ["GET", "POST"])
app.include_router(router)
