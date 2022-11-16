# *********************************************************************************************************************
# Program Name : response_vo
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
from pydantic import BaseModel, Extra


class BaseResponse(BaseModel):
    """
    Define pydantic model to validate basic output response
    """
    CODE: str
    ERROR_MSG: str

    class Config:
        extra = Extra.forbid


class MessageResponse(BaseResponse):
    """
    Define pydantic model to validate output response
    """
    MSG: str | dict

    class Config:
        extra = Extra.forbid


class PathResponse(BaseResponse):
    """
    Define pydantic model to validate output response
    """
    PATH: str

    class Config:
        extra = Extra.forbid


class TrainProgress(BaseResponse):
    """
    Define pydantic model to validate output response
    """
    MDL_LRNG_ST_CD: dict
    TRAIN_INFO: dict

    class Config:
        extra = Extra.forbid


class TrainResult(BaseResponse):
    """
    Define pydantic model to validate output response
    """
    RSLT_MSG: dict

    class Config:
        extra = Extra.forbid


class PredictResponse(BaseResponse):
    """
    Define pydantic model to validate output response
    """
    EVNT_ID: list
    PRBT: list

    class Config:
        extra = Extra.forbid


class DeployState(BaseResponse):
    """
    Define pydantic model to validate output response
    """
    DEPLOY_STATE: list
    CURRENT_DEPLOY_NUM: int
    MAX_DEPLOY: int

    class Config:
        extra = Extra.forbid
