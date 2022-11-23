# *********************************************************************************************************************
# Program Name : request_vo
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
from pydantic import BaseModel, Extra


class CheckTrainProgress(BaseModel):
    """
    Define pydantic model to validate inputs of train_progress request
    """
    MDL_ID: str
    MN_VER: str
    N_VER: str

    class Config:
        extra = Extra.forbid


class MakeDataset(BaseModel):
    """
    Define pydantic model to validate inputs of make_dataset request
    """
    MDL_ID: str
    MN_VER: str
    N_VER: str
    STYMD: str
    EDYMD: str
    LRNG_DATA_TGT_NCNT: str
    USR_ID: str

    class Config:
        extra = Extra.forbid


class Deploy(BaseModel):
    """
    Define pydantic model to validate inputs of deploy request
    """
    MDL_ID: str
    MN_VER: str
    N_VER: str
    MDL_TYP: int
    WDTB_SRVR_NCNT: int

    class Config:
        extra = Extra.forbid


class Predict(BaseModel):
    """
    Define pydantic model to validate inputs of predict request
    """
    MDL_ID: str
    MN_VER: str
    N_VER: str
    EVNT_THRU_PATH: list

    class Config:
        extra = Extra.forbid


class EndDeploy(BaseModel):
    """
    Define pydantic model to validate inputs of end_deploy request
    """
    model_id: str
    version: str

    class Config:
        extra = Extra.forbid


class Train(BaseModel):
    """
    Define pydantic model to validate inputs of train request
    """
    MDL_ID: str
    MN_VER: str
    N_VER: str
    EPOCH: int
    DATA_SPLIT: str
    EARLY_STOP: str
    BATCH_SIZE: int
    USR_ID: str

    class Config:
        extra = Extra.forbid


class StopTrain(BaseModel):
    """
    Define pydantic model to validate inputs of stop_train request
    """
    MDL_ID: str
    MN_VER: str
    N_VER: str

    class Config:
        extra = Extra.forbid


class BasicModelInfo(BaseModel):
    """
    Define pydantic model to validate inputs of basic request
    """
    MDL_ID: str
    MN_VER: str
    N_VER: str

    class Config:
        extra = Extra.forbid
