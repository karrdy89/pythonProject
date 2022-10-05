import json
import os
import logging
import uuid
from datetime import datetime, timedelta
from shutil import rmtree

import ray
import httpx
import tensorflow
from fastapi import FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from fastapi.responses import FileResponse
from sklearn.utils import resample
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask
from apscheduler.schedulers.background import BackgroundScheduler

import VO.request_vo as req_vo
import VO.response_vo as res_vo
import statics
from distribution.data_loader_nbo.data_loader import MakeDatasetNBO
from pipeline import Pipeline
from tensorboard_service import TensorBoardTool
from utils.common import version_encode
from pipeline import TrainInfo
from statics import Actors, TrainStateCode

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
        self._EXPIRE_TIME: int = 3600
        self._dataset_url: dict[str, str] = {}
        self._scheduler = BackgroundScheduler()
        self._scheduler.start()

    @router.post("/train/run")
    async def train(self, request_body: req_vo.Train):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: train run")
        model = request_body.MDL_NM
        main_version = str(request_body.MN_VER)
        sub_version = str(request_body.N_VER)
        version = main_version + '.' + sub_version
        pipeline_name = model + ":" + version
        if await self._shared_state.is_actor_exist.remote(name=pipeline_name):
            return json.dumps({"CODE": "FAIL", "ERROR_MSG": "same model is already running"})

        try:
            pipeline_actor = Pipeline.options(name="pipeline_name").remote()
        except ValueError as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="train run: failed to make actor: " + exc.__str__())
            return json.dumps({"CODE": "FAIL", "ERROR_MSG": "same process is already running"})
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="train run: failed to make actor: " + exc.__str__())
            return json.dumps({"CODE": "FAIL", "ERROR_MSG": "failed to create process"})

        set_shared_result = await self._shared_state.set_actor.remote(name=pipeline_name, act=pipeline_actor)
        if set_shared_result == 0:
            set_pipe_result = await pipeline_actor.set_pipeline.remote(name=model, version=version)
            if set_pipe_result == 0:
                train_info = TrainInfo()
                train_info.name = pipeline_name
                train_info.epoch = request_body.EPOCH
                train_info.early_stop = request_body.EARLY_STOP
                train_info.data_split = request_body.DATA_SPLIT
                train_info.batch_size = request_body.BATCH_SIZE
                tmp_path = model + "/" + str(version_encode(version))
                train_info.save_path = project_path + '/saved_models/' + tmp_path
                train_info.log_path = project_path + '/train_logs/' + tmp_path

                pipeline_actor.trigger_pipeline.remote(train_info=train_info)
                self._shared_state.set_train_status.remote(name=pipeline_name,
                                                           status_code=TrainStateCode.TRAINING)
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="train run: pipeline stated: " + pipeline_name)
                return json.dumps({"CODE": "SUCCESS", "ERROR_MSG": ""})
            else:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="train run: failed to set pipeline: " + pipeline_name)
                return json.dumps({"CODE": "FAIL", "ERROR_MSG": "failed to set train process"})
        else:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="train run: max concurrent exceeded: " + pipeline_name)
            return json.dumps({"CODE": "FAIL", "ERROR_MSG": "max concurrent exceeded"})

    @router.post("/train/stop")
    async def stop_train(self, request_body: req_vo.StopTrain):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: train stop")
        model = request_body.MDL_NM
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = str(main_version) + "." + str(sub_version)
        pipeline_name = model + ":" + version
        await self._shared_state.set_train_status.remote(name=pipeline_name, status_code=TrainStateCode.TRAINING_FAIL)
        kill_actor_result = await self._shared_state.kill_actor.remote(name=pipeline_name)
        if kill_actor_result == 0:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="get request: train stopped : "+pipeline_name)
            return json.dumps({"CODE": "SUCCESS", "ERROR_MSG": ""})
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="train stopped fail: model not exist: "+pipeline_name)
            return json.dumps({"CODE": "FAIL", "ERROR_MSG": "training process not exist"})

    @router.post("/train/progress")
    async def get_train_progress(self, request_body: req_vo.CheckTrainProgress):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: train progress")
        model = request_body.MDL_NM
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = str(main_version) + '.' + str(sub_version)
        name = model + ":" + version
        train_state = await self._shared_state.get_status_code.remote(name=name)
        pipeline_state = await self._shared_state.get_pipeline_result.remote(name=name)
        train_result = await self._shared_state.get_train_result.remote(name=name)
        result = res_vo.RstCheckTrainProgress(MDL_LRNG_ST_CD=train_state,
                                              CODE="SUCCESS",
                                              ERROR_MSG="",
                                              TRAIN_INFO={"pipline_state": pipeline_state,
                                                          "train_result": train_result})
        return result.json()

    @router.post("/train/result")
    async def get_train_result(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: get train result")
        model = request_body.MDL_NM
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = str(main_version) + '.' + str(sub_version)
        name = model + ":" + version
        try:
            train_result = await self._shared_state.get_train_result.remote(name=name)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="get request: an error occur while get train result: " + exc.__str__())
            return json.dumps({"CODE": "FAIL", "ERROR_MSG": "INTERNAL ERROR", "RSLT_MSG": ""})
        else:
            if len(train_result) == 0:
                return json.dumps({"CODE": "FAIL", "ERROR_MSG": "model not exist", "RSLT_MSG": ""})
            else:
                total_result = {"train_result": train_result["train_result"],
                                "test_result": train_result["test_result"]}
                return json.dumps({"CODE": "SUCCESS", "ERROR_MSG": "", "RSLT_MSG": total_result})

    @router.post("/dataset/make")
    async def make_dataset(self, request_body: req_vo.MakeDataset):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: make dataset")
        model = request_body.MDL_NM
        if model not in statics.MODELS:
            return json.dumps({"CODE": "FAIL", "ERROR_MSG": "MODEL NOT FOUND"})
        main_version = str(request_body.MN_VER)
        sub_version = str(request_body.N_VER)
        name = model + ":" + main_version + '.' + sub_version
        start_dtm = request_body.STYMD
        end_dtm = request_body.EDYMD
        if model == "NBO":
            try:
                dataset_maker = MakeDatasetNBO.options(name=name).remote()
            except ValueError as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="make dataset: failed to make actor MakeDatasetNBO: " + exc.__str__())
                return json.dumps({"CODE": "FAIL", "ERROR_MSG": "same process is already running"})
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="make dataset: failed to make actor MakeDatasetNBO: " + exc.__str__())
                return json.dumps({"CODE": "FAIL", "ERROR_MSG": "failed to create process"})
            else:
                if await self._shared_state.is_actor_exist.remote(name=name):
                    return json.dumps({"CODE": "FAIL", "ERROR_MSG": "same task is already running"})

            labels = ["EVT000", "EVT100", "EVT200", "EVT300", "EVT400", "EVT500", "EVT600", "EVT700", "EVT800",
                      "EVT900"]
            key_index = 0
            x_index = [1]
            version = sub_version
            num_data_limit = int(request_body.LRNG_DATA_TGT_NCNT)
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="make dataset: init MakeDatasetNBO")
            result = await dataset_maker.init.remote(name=name, act=dataset_maker, labels=labels, version=version,
                                                     key_index=key_index, x_index=x_index,
                                                     num_data_limit=num_data_limit,
                                                     start_dtm=start_dtm, end_dtm=end_dtm)
            if result == 0:
                set_shared_result = await self._shared_state.set_actor.remote(name=name, act=dataset_maker)
                if set_shared_result == 0:
                    self._shared_state.set_make_dataset_result.remote(name=name,
                                                                      state_code=TrainStateCode.MAKING_DATASET)
                    dataset_maker.fetch_data.remote()
                    self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                            msg="make dataset: running")
                    return json.dumps({"CODE": "SUCCESS", "ERROR_MSG": ""})
                else:
                    return json.dumps({"CODE": "FAIL", "ERROR_MSG": "max concurrent exceeded"})
            else:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="make dataset: failed to init MakeDatasetNBO")
                return json.dumps({"CODE": "FAIL", "ERROR_MSG": "failed to init process"})

    @router.get("/dataset/download/{dataset_name}/{version}")
    async def get_dataset_url(self, dataset_name: str, version: str):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: download dataset url")
        path = statics.ROOT_DIR+"/dataset/"+dataset_name+"/"+version+"/"+"dataset_"+dataset_name+"_"+version+".zip"
        if not os.path.exists(path):
            return "file not exist"
        uid = str(uuid.uuid4())
        run_date = datetime.now() + timedelta(seconds=self._EXPIRE_TIME)
        self._scheduler.add_job(self.expire_dataset_url, "date", run_date=run_date, args=[uid])
        self._dataset_url[uid] = path
        return "/dataset/" + uid

    @router.get("/dataset/{uid}")
    async def download_dataset(self, uid: str):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: download dataset")
        if uid in self._dataset_url:
            path = self._dataset_url[uid]
            if os.path.exists(path):
                self.expire_dataset_url(uid)
                return FileResponse(path)
            else:
                return "file not exist"
        else:
            return "invalid url"

    @router.delete("/dataset/download/{dataset_name}/{version}")
    async def delete_dataset(self, dataset_name: str, version: str):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: delete dataset")
        path = statics.ROOT_DIR + "/dataset/" + dataset_name + "/" + version
        if not os.path.exists(path):
            return "file not exist"
        else:
            try:
                rmtree(path)
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="an error occur when delete dataset: " + exc.__str__())
                return "failed to delete dataset"
            else:
                return "deleted"

    @router.post("/deploy")
    async def deploy(self, request_body: req_vo.Deploy):
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
    async def add_container(self, request_body: req_vo.AddContainer):
        remote_job_obj = self._server.add_container.remote(model_id=request_body.model_id, version=request_body.version,
                                                           container_num=request_body.container_num)
        result = await remote_job_obj
        return result

    @router.post("/deploy/remove_container")
    async def remove_container(self, request_body: req_vo.RemoveContainer):
        remote_job_obj = self._server.remove_container.remote(model_id=request_body.model_id,
                                                              version=request_body.version,
                                                              container_num=request_body.container_num)
        result = await remote_job_obj
        return result

    @router.post("/deploy/end_deploy")
    async def end_deploy(self, request_body: req_vo.EndDeploy):
        remote_job_obj = self._server.end_deploy.remote(model_id=request_body.model_id, version=request_body.version)
        result = await remote_job_obj
        return result

    @router.post("/predict")
    async def predict(self, request_body: req_vo.Predict):
        remote_job_obj = self._server.predict.remote(model_id=request_body.model_id, version=request_body.version,
                                                     data=request_body.feature)
        result = await remote_job_obj
        return result

    @router.post("/tensorboard")
    async def create_tensorboard(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: create tensorboard")
        model = request_body.MDL_NM
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = str(main_version) + "." + str(sub_version)
        encoded_version = version_encode(version)
        log_path = project_path + "/train_logs/" + model + "/" + str(encoded_version)
        if os.path.isdir(log_path):
            port = self._tensorboard_tool.run(dir_path=log_path)
            path = "/tensorboard/" + str(port)
            return json.dumps({"CODE": "SUCCESS", "ERROR_MSG": "", "PATH": path})
        else:
            return json.dumps({"CODE": "FAIL", "ERROR_MSG": "log file not exist", "PATH": ""})

    def expire_dataset_url(self, uid) -> None:
        if uid in self._dataset_url:
            del self._dataset_url[uid]
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="dataset url has expired: " + uid)


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
