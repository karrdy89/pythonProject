from pydantic import BaseModel, Extra


class BaseResponse(BaseModel):
    CODE: str
    ERROR_MSG: str

    class Config:
        extra = Extra.forbid


class MessageResponse(BaseResponse):
    MSG: str | dict

    class Config:
        extra = Extra.forbid


class PathResponse(BaseResponse):
    PATH: str

    class Config:
        extra = Extra.forbid


class TrainProgress(BaseResponse):
    MDL_LRNG_ST_CD: dict
    TRAIN_INFO: dict

    class Config:
        extra = Extra.forbid


class TrainResult(BaseResponse):
    RSLT_MSG: dict

    class Config:
        extra = Extra.forbid


class PredictResponse(BaseResponse):
    EVNT_ID: list
    PRBT: list

    class Config:
        extra = Extra.forbid


class DeployState(BaseResponse):
    DEPLOY_STATE: list
    CURRENT_DEPLOY_NUM: int

    class Config:
        extra = Extra.forbid
