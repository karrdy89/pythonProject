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
import os
import logging
import uuid
import yaml

import ray
from fastapi import FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.cors import CORSMiddleware

import VO.request_vo as req_vo
import VO.response_vo as res_vo
from dataset_maker.constructor import construct_operator
from pipeline import Pipeline
from pipeline import SequenceNotExistError
from utils import version_encode
from pipeline import TrainInfo
from statics import Actors, TrainStateCode, ROOT_DIR

project_path = ROOT_DIR
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
    _serving_manager : actor
        An actor handle of _serving_manager.
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
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: is_trainable" + str(dict(request_body)))
        model_id = request_body.MDL_ID
        psb_yn_dataset = "N"
        psb_yn_train = "N"
        path_dataset_def = ROOT_DIR + "/script/dataset_maker/dataset_definitions.yaml"
        path_pipeline_def = ROOT_DIR + "/script/pipeline/pipelines.yaml"

        with open(path_dataset_def, 'r') as stream:
            try:
                dataset_def = yaml.safe_load(stream)
            except Exception as exc:
                self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                        msg="get request: is_trainable: fail: can't read definition file"
                                            + str(exc) + str(dict(request_body)))
                return res_vo.IsTrainable(CODE="FAIL", ERROR_MSG="can't read definition file: "+exc.__str__(),
                                          DATASET_YN=psb_yn_dataset, TRAIN_YN=psb_yn_train)
            else:
                data_maker_list = dataset_def.get("dataset_definitions", '')
                if data_maker_list == '':
                    self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                            msg="get request: is_trainable: fail: there is no dataset_definitions: "
                                                + str(dict(request_body)))
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
                self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                        msg="get request: is_trainable: fail: can't read definition file"
                                            + str(exc) + str(dict(request_body)))
                return res_vo.IsTrainable(CODE="FAIL", ERROR_MSG="can't read definition file: "+exc.__str__(),
                                          DATASET_YN=psb_yn_dataset, TRAIN_YN=psb_yn_train)
            else:
                pipeline_list = pipeline_def.get("pipelines", '')
                if pipeline_list == '':
                    self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                            msg="there is no pipeline: " + self._name + str(dict(request_body)))
                    return res_vo.IsTrainable(CODE="FAIL", ERROR_MSG="definitions not exist: ",
                                              DATASET_YN=psb_yn_dataset, TRAIN_YN=psb_yn_train)
                for pipeline in pipeline_list:
                    if pipeline.get("name") == model_id:
                        psb_yn_train = "Y"
                        break
        return res_vo.IsTrainable(CODE="SUCCESS", ERROR_MSG="", DATASET_YN=psb_yn_dataset, TRAIN_YN=psb_yn_train)

    @router.post("/train/run", response_model=res_vo.BaseResponse)
    async def train(self, request_body: req_vo.Train):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: train: " + str(dict(request_body)))
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
                                    msg="get request: train: failed to make actor: " + exc.__str__()
                                        + str(dict(request_body)))
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="same process is already running")
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="get request: train: failed to make actor: " + exc.__str__()
                                        + str(dict(request_body)))
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="failed to create process")

        set_shared_result = await self._shared_state.set_actor.remote(name=pipeline_name, act=pipeline_actor)
        if set_shared_result == 0:
            try:
                await pipeline_actor.set_pipeline.remote(model_id=model_id, version=version, user_id=user_id)
            except SequenceNotExistError as exc:
                self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                        msg="get request: train: train pipeline not exist on this model : "
                                            + exc.__str__() + str(dict(request_body)))
                return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="train pipeline not exist on this model")
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="get request: train: an error occur when set train pipeline : "
                                            + exc.__str__() + str(dict(request_body)))
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
                                        msg="train run: pipeline stated: " + str(dict(request_body)))
                return res_vo.BaseResponse(CODE="SUCCESS", ERROR_MSG="")
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="train run: fail: max concurrent exceeded: " + str(dict(request_body)))
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="max concurrent exceeded")

    @router.post("/train/stop", response_model=res_vo.BaseResponse)
    async def stop_train(self, request_body: req_vo.StopTrain):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: stop_train :" + str(dict(request_body)))
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + "." + sub_version
        pipeline_name = model_id + ":" + version

        actor = await self._shared_state.get_actor.remote(name=pipeline_name)
        if actor is None:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="get request: stop_train: fail: training process not exist"
                                        + str(dict(request_body)))
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="training process not exist")
        res = await actor.kill_process.remote()
        if res == 0:
            try:
                self._shared_state.kill_actor.remote(pipeline_name)
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="stop_train: fail: failed to kill actor: "
                                            + exc.__str__() + str(dict(request_body)))
                return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="training process not exist")
            else:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="get request: stop_train: success : " + str(dict(request_body)))
                return res_vo.BaseResponse(CODE="SUCCESS", ERROR_MSG="")
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                    msg="stop_train: fail: model not exist: " + str(dict(request_body)))
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="training process not exist")

    @router.post("/train/progress", response_model=res_vo.TrainProgress)
    async def get_train_progress(self, request_body: req_vo.CheckTrainProgress):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: get_train_progress: " + str(dict(request_body)))
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
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get_train_progress: success" + str(dict(request_body)))
        return result

    @router.post("/train/result", response_model=res_vo.TrainResult)
    async def get_train_result(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: get_train_result: " + str(dict(request_body)))
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + '.' + sub_version
        name = model_id + ":" + version
        try:
            train_result = await self._shared_state.get_train_result.remote(name=name)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="get_train_result: fail: " + exc.__str__() + str(dict(request_body)))

            return res_vo.TrainResult(CODE="FAIL", ERROR_MSG=exc.__str__(), RSLT_MSG="")
        else:
            if len(train_result) == 0:
                self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                        msg="get_train_result: fail: model not exist" + str(dict(request_body)))
                return res_vo.TrainResult(CODE="FAIL", ERROR_MSG="model not exist", RSLT_MSG="")
            else:
                total_result = {"train_result": train_result["train_result"],
                                "test_result": train_result["test_result"]}
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="get_train_result: success" + str(dict(request_body)))
                return res_vo.TrainResult(CODE="SUCCESS", ERROR_MSG="", RSLT_MSG=total_result)

    @router.post("/dataset/make", response_model=res_vo.BaseResponse)
    async def make_dataset(self, request_body: req_vo.MakeDataset):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: make_dataset: " + str(dict(request_body)))
        try:
            operator_args = construct_operator(request_body)
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="get request: make_dataset: fail: failed to construct actor: "
                                        + exc.__str__() + str(dict(request_body)))
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="failed to construct actor: " + exc.__str__())
        else:
            actor = operator_args.actor_handle
            actor_name = operator_args.actor_name
            if await self._shared_state.is_actor_exist.remote(name=operator_args.actor_name):
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="get request: make_dataset: fail: same task is already running: "
                                            + str(dict(request_body)))
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
                                                msg="get request: make_dataset: successes" + str(dict(request_body)))
                        return res_vo.BaseResponse(CODE="SUCCESS", ERROR_MSG="")
                    else:
                        ray.kill(actor)
                        self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                                msg="get request: make_dataset: fail: max concurrent exceeded"
                                                    + str(dict(request_body)))
                        return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="max concurrent exceeded")
                else:
                    self._logger.log.remote(level=logging.WARN, worker=self._worker,
                                            msg="get request: make_dataset: failed to init MakeDatasetNBO"
                                                + str(dict(request_body)))
                    ray.kill(actor)
                    return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="failed to init process")

    @router.post("/dataset/stop", response_model=res_vo.BaseResponse)
    async def dataset_make_stop(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: dataset_make_stop: " + str(dict(request_body)))
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        name = model_id + ":" + main_version + '.' + sub_version
        act = await self._shared_state.get_actor.remote(name=name)
        if act is not None:
            await self._shared_state.set_error_message.remote(name=name, msg="interruption due to stop request")
            kill_result = await act.kill_process.remote()
            if kill_result == 0:
                self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="dataset_make_stop: success" + str(dict(request_body)))
                return res_vo.BaseResponse(CODE="SUCCESS", ERROR_MSG="")
            else:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                        msg="dataset_make_stop: failed to kill actor: " + str(dict(request_body)))
                return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="failed to kill task")
        else:
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="dataset_make_stop: fail: task not found" + str(dict(request_body)))
            return res_vo.BaseResponse(CODE="FAIL", ERROR_MSG="task not found")

    @router.post("/dataset/download", response_model=res_vo.PathResponse)
    async def get_dataset_url(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: get_dataset_url: " + str(dict(request_body)))
        dataset_name = request_body.MDL_ID
        version = request_body.MN_VER
        path = ROOT_DIR + "/dataset/" + dataset_name + "/" + version + "/" + dataset_name + "_" + version + ".zip"
        if not os.path.exists(path):
            self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                    msg="get request: get_dataset_url:: fail: dataset not exist" + str(dict(request_body)))
            return res_vo.PathResponse(CODE="FAIL", ERROR_MSG="dataset not exist", PATH="")
        uid = str(uuid.uuid4())
        await self._shared_state.set_dataset_url.remote(uid=uid, path=path)
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: get_dataset_url:: success" + str(dict(request_body)))
        return res_vo.PathResponse(CODE="SUCCESS", ERROR_MSG="", PATH="/dataset/" + uid)

    @router.post("/deploy", response_model=res_vo.MessageResponse)
    async def deploy(self, request_body: req_vo.Deploy):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: deploy: " + str(dict(request_body)))
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
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: deploy: success" + str(dict(request_body)))
        return result

    @router.get("/deploy/state", response_model=res_vo.DeployState)
    async def get_deploy_state(self):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: get_deploy_state")
        return await self._serving_manager.get_deploy_state.remote()

    @router.post("/deploy/end_deploy", response_model=res_vo.MessageResponse)
    async def end_deploy(self, request_body: req_vo.BasicModelInfo):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: end_deploy: " + str(dict(request_body)))
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + '.' + sub_version
        result = await self._serving_manager.end_deploy.remote(model_id=model_id, version=version)
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: end_deploy: success" + str(dict(request_body)))
        return result

    @router.post("/predict", response_model=res_vo.PredictResponse)
    async def predict(self, request_body: req_vo.Predict):
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="get request: predict: " + str(dict(request_body)))
        model_id = request_body.MDL_ID
        main_version = request_body.MN_VER
        sub_version = request_body.N_VER
        version = main_version + "." + sub_version
        data = request_body.INPUT_DATA
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


app.include_router(router)
