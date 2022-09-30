import json
import os
import logging

import ray
import httpx
import tensorflow
from fastapi import FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from fastapi.responses import FileResponse
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

import VO.request_vo as rvo
import statics
from distribution.data_loader_nbo.data_loader import MakeDatasetNBO
from pipeline import Pipeline
from tensorboard_service import TensorBoardTool
from utils.common import version_encode
from pipeline import TrainInfo
from statics import Actors

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
        self._server: ray.actor = ray.get_actor(Actors.MODEL_SERVER)
        self._logger: ray.actor = ray.get_actor(Actors.LOGGER)
        self._shared_state: ray.actor = ray.get_actor(Actors.GLOBAL_STATE)
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

    @router.post("/train/state")
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
    async def make_dataset(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: make dataset")
        try:
            dataset_maker = MakeDatasetNBO.options(name=Actors.DATA_MAKER_NBO).remote()
        except Exception as exc:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="make dataset: failed to make actor MakeDatasetNBO: " + exc.__str__())
            return "make dataset fail"
        else:
            if ray.get(self._shared_state.is_actor_exist.remote(name=Actors.DATA_MAKER_NBO)):
                return "the task is already running"
            self._shared_state.set_actor.remote(name=Actors.DATA_MAKER_NBO, act=dataset_maker)

        labels = ["EVT000", "EVT100", "EVT200", "EVT300", "EVT400", "EVT500", "EVT600", "EVT700", "EVT800",
                  "EVT900"]  # input
        key_index = 0  # input
        x_index = [1]  # input
        version = '0'  # input
        num_data_limit = 100000
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="make dataset: init MakeDatasetNBO")
        result = await dataset_maker.init.remote(act=dataset_maker, labels=labels, version=version,
                                                 key_index=key_index, x_index=x_index,
                                                 num_data_limit=num_data_limit)
        if result == 0:
            dataset_maker.fetch_data.remote()
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="make dataset: running")
            return "make dataset start"
        else:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="make dataset: failed to init MakeDatasetNBO")
            return "make dataset fail"

    @router.get("/dataset/download_url")
    async def get_dataset_url(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: download dataset url")
        # create with uuid, if download->expire,
        # id download check valid path, if valid download if not reject
        pass

    @router.get("/dataset/download/{dataset_name}/{version}")
    async def download_dataset(self, dataset_name: str, version: str):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: download dataset")
        path = statics.ROOT_DIR+"/dataset/"+dataset_name+"/"+version+"/"+"dataset_"+dataset_name+"_"+version+".zip"
        if os.path.exists(path):
            return FileResponse(path)
        else:
            return "file not exist"

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

    @router.get("/db")
    async def db_test(self):
        from db.db_util import DBUtil
        db = DBUtil()
        result = db.select("test")
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
