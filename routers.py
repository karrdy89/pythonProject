# *********************************************************************************************************************
# Program Name : routers
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import ast
import os
import logging
import uuid
import requests
import json
import yaml

import ray
import httpx
from fastapi import FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

import VO.request_vo as req_vo
import VO.response_vo as res_vo
import statics
from dataset_maker.constructor import construct_operator
from pipeline import Pipeline
from pipeline.exceptions import SequenceNotExistError
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
    _logger : actor
        An actor handle of global logger.
    _tf_serving_manager : actor
        An actor handle of tf_serving_manager.
    _onx_serving_manager : actor
        An actor handle of onx_serving_manager.
    _shared_state : actor
        An actor handle of global state manager.

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    @router:
        API endpoints of server
    """

    def __init__(self):
        self._worker = type(self).__name__
        self._logger: ray.actor = ray.get_actor(Actors.LOGGER)
        self._serving_manager: ray.actor = ray.get_actor(Actors.SERVING_MANAGER)
        self._shared_state: ray.actor = ray.get_actor(Actors.GLOBAL_STATE)

    @router.post("/dataset/psbYn", response_model=res_vo.IsTrainable)
    async def is_trainable(self, request_body: req_vo.ModelID):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: is_trainable")
        model_id = request_body.MDL_ID
        psb_yn_dataset = "N"
        psb_yn_train = "N"
        path_dataset_def = statics.ROOT_DIR + "/dataset_maker/dataset_definitions.yaml"
        path_pipeline_def = statics.ROOT_DIR + "/pipeline/pipelines.yaml"

        with open(path_dataset_def, 'r') as stream:
            try:
                dataset_def = yaml.safe_load(stream)
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg=str(exc))
                return res_vo.IsTrainable(CODE="FAIL", ERROR_MSG="can't read definition file: "+exc.__str__(),
                                          DATASET_YN=psb_yn_dataset, TRAIN_YN=psb_yn_train)
            else:
                data_maker_list = dataset_def.get("dataset_definitions", '')
                if data_maker_list == '':
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                            msg="there is no dataset_definitions: " + self._name)
                    return res_vo.IsTrainable(CODE="FAIL", ERROR_MSG="definitions not exist: ",
                                              DATASET_YN=psb_yn_dataset, TRAIN_YN=psb_yn_train)
                for data_maker in data_maker_list:
                    if data_maker.get("name") == model_id:
                        psb_yn_dataset = "Y"
                        break

        with open(path_pipeline_def, 'r') as stream:
            try:
                pipeline_def = yaml.safe_load(stream)
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg=str(exc))
                return res_vo.IsTrainable(CODE="FAIL", ERROR_MSG="can't read definition file: "+exc.__str__(),
                                          DATASET_YN=psb_yn_dataset, TRAIN_YN=psb_yn_train)
            else:
                pipeline_list = pipeline_def.get("pipelines", '')
                if pipeline_list == '':
                    self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                            msg="there is no pipeline: " + self._name)
                    return res_vo.IsTrainable(CODE="FAIL", ERROR_MSG="definitions not exist: ",
                                              DATASET_YN=psb_yn_dataset, TRAIN_YN=psb_yn_train)
                for pipeline in pipeline_list:
                    if pipeline.get("name") == model_id:
                        psb_yn_train = "Y"
                        break
        return res_vo.IsTrainable(CODE="SUCCESS", ERROR_MSG="", DATASET_YN=psb_yn_dataset, TRAIN_YN=psb_yn_train)

    @router.post("/train/run", response_model=res_vo.BaseResponse)
    async def train(self, request_body: req_vo.Train):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: train")
        model_id = request_body.MDL_ID
        user_id = request_body.USR_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + '.' + sub_version
        pipeline_name = model_id + ":" + version
        if await self._shared_state.is_actor_exist.remote(name=pipeline_name):
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="same model is already running")
        try:
            pipeline_actor = Pipeline.options(name="pipeline_name", max_concurrency=5).remote()
        except ValueError as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="train: failed to make actor: " + exc.__str__())
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="same process is already running")
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="train: failed to make actor: " + exc.__str__())
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="failed to create process")

        set_shared_result = await self._shared_state.set_actor.remote(name=pipeline_name, act=pipeline_actor)
        if set_shared_result == 0:
            try:
                await pipeline_actor.set_pipeline.remote(model_id=model_id, version=version, user_id=user_id)
            except SequenceNotExistError as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="train: train pipeline not exist on this model : " + exc.__str__())
                return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="train pipeline not exist on this model")
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="train: an error occur when set train pipeline : " + model_id + exc.__str__())
                return res_vo.BaseResponse(CODE="FAIL",
                                           ERROR_MSG="an error occur when set train pipeline : " + model_id)
            else:
                train_info = TrainInfo()
                train_info.name = pipeline_name
                train_info.epoch = request_body.EPOCH
                train_info.early_stop = request_body.EARLY_STOP
                train_info.data_split = request_body.DATA_SPLIT
                train_info.batch_size = request_body.BATCH_SIZE
                tmp_path = model_id + "/" + str(version_encode(version))
                model_key = model_id + "_" + version
                train_info.save_path = project_path + '/saved_models/' + model_key + "/" + tmp_path
                train_info.log_path = project_path + '/train_logs/' + tmp_path
                pipeline_actor.trigger_pipeline.remote(train_info=train_info)
                self._shared_state.set_train_status.remote(name=pipeline_name, user_id=user_id,
                                                           status_code=TrainStateCode.TRAINING)
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="train run: pipeline stated: " + pipeline_name)
                return res_vo.BaseResponse(CODE="SUCCESS", ERROR_MSG="")
        else:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="train run: max concurrent exceeded: " + pipeline_name)
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="max concurrent exceeded")

    @router.post("/train/stop", response_model=res_vo.BaseResponse)
    async def stop_train(self, request_body: req_vo.StopTrain):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: stop_train")
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + "." + sub_version
        pipeline_name = model_id + ":" + version

        actor = await self._shared_state.get_actor.remote(name=pipeline_name)
        res = await actor.kill_process.remote()
        if res == 0:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="stop_train: success : " + pipeline_name)
            return res_vo.BaseResponse(CODE="SUCCESS", ERROR_MSG="")
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="stop_train: fail: model not exist: " + pipeline_name)
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="training process not exist")

        # await self._shared_state.set_error_message.remote(name=pipeline_name, msg="interruption due to stop request")
        # kill_actor_result = await self._shared_state.kill_actor.remote(name=pipeline_name)
        # if kill_actor_result == 0:
        #     self._logger.log.remote(level=logging.INFO, worker=self._worker,
        #                             msg="stop_train: success : " + pipeline_name)
        #     return res_vo.BaseResponse(CODE="SUCCESS", ERROR_MSG="")
        # else:
        #     self._logger.log.remote(level=logging.WARN, worker=self._worker,
        #                             msg="stop_train: fail: model not exist: " + pipeline_name)
        #     return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="training process not exist")

    @router.post("/train/progress", response_model=res_vo.TrainProgress)
    async def get_train_progress(self, request_body: req_vo.CheckTrainProgress):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: get_train_progress")
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + '.' + sub_version
        name = model_id + ":" + version
        train_state = await self._shared_state.get_status_code.remote(name=name)
        pipeline_state = await self._shared_state.get_pipeline_result.remote(name=name)
        train_result = await self._shared_state.get_train_result.remote(name=name)
        error_message = await self._shared_state.get_error_message.remote(name=name)
        result = res_vo.TrainProgress(MDL_LRNG_ST_CD=train_state,
                                      CODE="SUCCESS",
                                      ERROR_MSG="",
                                      TRAIN_INFO={"pipline_state": pipeline_state,
                                                  "train_result": train_result,
                                                  "error_msg": error_message})
        return result

    @router.post("/train/result", response_model=res_vo.TrainResult)
    async def get_train_result(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: get_train_result")
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + '.' + sub_version
        name = model_id + ":" + version
        try:
            train_result = await self._shared_state.get_train_result.remote(name=name)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="get_train_result: fail: " + exc.__str__())

            return res_vo.TrainResult(CODE="FAIL", ERROR_MSG=exc.__str__(), RSLT_MSG="")
        else:
            if len(train_result) == 0:
                return res_vo.TrainResult(CODE="FAIL", ERROR_MSG="model not exist", RSLT_MSG="")
            else:
                total_result = {"train_result": train_result["train_result"],
                                "test_result": train_result["test_result"]}
                return res_vo.TrainResult(CODE="SUCCESS", ERROR_MSG="", RSLT_MSG=total_result)

    @router.post("/dataset/make", response_model=res_vo.BaseResponse)
    async def make_dataset(self, request_body: req_vo.MakeDataset):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: make_dataset")
        try:
            operator_args = construct_operator(request_body)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="make_dataset: failed to construct actor: " + exc.__str__())
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="failed to construct actor: " + exc.__str__())
        else:
            actor = operator_args.actor_handle
            actor_name = operator_args.actor_name
            if await self._shared_state.is_actor_exist.remote(name=operator_args.actor_name):
                return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="same task is already running")
            else:
                result = await actor.init.remote(args=operator_args)
                if result == 0:
                    set_shared_result = await self._shared_state.set_actor.remote(name=actor_name, act=actor)
                    if set_shared_result == 0:
                        self._shared_state.set_make_dataset_result.remote(name=actor_name, user_id=request_body.USR_ID,
                                                                          state_code=TrainStateCode.MAKING_DATASET)
                        actor.fetch_data.remote()
                        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                                msg="make_dataset: running")
                        return res_vo.BaseResponse(CODE="SUCCESS", ERROR_MSG="")
                    else:
                        ray.kill(actor)
                        return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="max concurrent exceeded")
                else:
                    self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                            msg="make_dataset: failed to init MakeDatasetNBO")
                    ray.kill(actor)
                    return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="failed to init process")

    @router.post("/dataset/stop", response_model=res_vo.BaseResponse)
    async def dataset_make_stop(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: dataset_make_stop")
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        name = model_id + ":" + main_version + '.' + sub_version
        act = await self._shared_state.get_actor.remote(name=name)
        if act is not None:
            await self._shared_state.set_error_message.remote(name=name, msg="interruption due to stop request")
            kill_result = await act.kill_process.remote()
            if kill_result == 0:
                return res_vo.BaseResponse(CODE="SUCCESS", ERROR_MSG="")
            else:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="dataset_make_stop: failed to kill actor")
                return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="failed to kill task")
        else:
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="task not found")

    @router.post("/dataset/download", response_model=res_vo.PathResponse)
    async def get_dataset_url(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: get_dataset_url")
        dataset_name = request_body.MDL_ID
        version = request_body.MN_VER
        path = statics.ROOT_DIR + "/dataset/" + dataset_name + "/" + version + "/" + dataset_name + "_" + version + ".zip"
        if not os.path.exists(path):
            return res_vo.PathResponse(CODE="FAIL", ERROR_MSG="dataset not exist", PATH="")
        uid = str(uuid.uuid4())
        await self._shared_state.set_dataset_url.remote(uid=uid, path=path)
        return res_vo.PathResponse(CODE="SUCCESS", ERROR_MSG="", PATH="/dataset/" + uid)

    @router.get("/dataset/{uid}")
    async def download_dataset(self, uid: str):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: download_dataset")
        path = await self._shared_state.get_dataset_path.remote(uid=uid)
        if path is not None:
            if os.path.exists(path):
                filename = path.split("/")[-1]
                return FileResponse(path, filename=filename)
            else:
                return "file not exist"
        else:
            return "invalid url"

    @router.post("/deploy", response_model=res_vo.MessageResponse)
    async def deploy(self, request_body: req_vo.Deploy):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: deploy")
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + '.' + sub_version
        model_type = request_body.MDL_TY_CD
        result = await self._serving_manager.deploy.remote(model_id=model_id,
                                                           version=version,
                                                           deploy_num=request_body.WDTB_SRVR_NCNT,
                                                           model_type=model_type)
        result = res_vo.MessageResponse.parse_obj(result)
        return result

    @router.get("/deploy/state", response_model=res_vo.DeployState)
    async def get_deploy_state(self):
        return await self._serving_manager.get_deploy_state.remote()

    @router.post("/deploy/end_deploy", response_model=res_vo.MessageResponse)
    async def end_deploy(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: end_deploy")
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + '.' + sub_version
        result = await self._serving_manager.end_deploy.remote(model_id=model_id, version=version)
        return result

    @router.post("/predict", response_model=res_vo.PredictResponse)
    async def predict(self, request_body: req_vo.Predict):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: predict")
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + "." + sub_version
        data = request_body.EVNT_THRU_PATH
        result = await self._serving_manager.predict.remote(model_id=model_id, version=version,
                                                            data=data)
        return result

    @router.post("/tensorboard", response_model=res_vo.PathResponse)
    async def create_tensorboard(self, request_body: req_vo.CreateTensorBoard):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="get request: create_tensorboard")
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        session_id = request_body.SESSION_ID
        version = main_version + "." + sub_version
        encoded_version = version_encode(version)
        log_path = project_path + "/train_logs/" + model_id + "/" + str(encoded_version)
        if os.path.isdir(log_path):
            port = await self._shared_state.get_tensorboard_port.remote(dir_path=log_path, session_id=session_id)
            if port == -1:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="get request: create_tensorboard: failed to launch tensorboard")
                return res_vo.PathResponse(CODE="FAIL", ERROR_MSG="failed to launch tensorboard", PATH="")
            elif port == -2:
                self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                        msg="get request: create_tensorboard: max tensorboard thread exceeded")
                return res_vo.PathResponse(CODE="FAIL", ERROR_MSG="max tensorboard thread exceeded", PATH="")
            else:
                path = "/tensorboard/" + str(port)
                return res_vo.PathResponse(CODE="SUCCESS", ERROR_MSG="", PATH=path)
        else:
            return res_vo.PathResponse(CODE="FAIL", ERROR_MSG="train log not exist", PATH="")


async def _reverse_proxy(request: Request):
    path = request.url.path.split('/')
    port = path[2]
    path = '/'.join(path[3:])

    # validate session request to manage server
    shared_state = ray.get_actor(Actors.GLOBAL_STATE)
    session_info, url = await shared_state.get_session.remote(port=port)
    if session_info is None:
        return PlainTextResponse("invalid access")
    else:
        data = {"SESSION_ID": session_info}
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        try:
            res = requests.post(url, data=json.dumps(data), headers=headers, timeout=10)
        except Exception as exc:
            return PlainTextResponse("failed to validate session:" + exc.__str__())
        else:
            if res.status_code == 200:
                body = ast.literal_eval(res.content.decode('utf-8'))
                is_valid = body.get("IS_VALID_SESSION_ID")
                if is_valid == 'Y':
                    client = httpx.AsyncClient(base_url="http://127.0.0.1:" + port)
                    url = httpx.URL(path=path,
                                    query=request.url.query.encode("utf-8"))
                    rp_req = client.build_request(request.method, url,
                                                  headers=request.headers.raw,
                                                  content=await request.body())
                    try:
                        rp_resp = await client.send(rp_req, stream=True)
                    except httpx.RequestError:
                        return PlainTextResponse("invalid URL")
                    except Exception as exc:
                        return f"An error occurred while requesting {exc.__str__()!r}."
                    else:
                        return StreamingResponse(
                            rp_resp.aiter_raw(),
                            status_code=rp_resp.status_code,
                            headers=rp_resp.headers,
                            background=BackgroundTask(rp_resp.aclose),
                        )
                else:
                    return PlainTextResponse("invalid access")
            else:
                return PlainTextResponse("failed to validate session")


app.add_route("/dashboard/{port}/{path:path}", _reverse_proxy, ["GET", "POST"])
app.add_route("/tensorboard/{port}/{path:path}", _reverse_proxy, ["GET", "POST"])
app.include_router(router)
