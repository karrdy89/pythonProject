from pydantic import BaseModel, Extra


class CheckTrainProgress(BaseModel):
    MDL_ID: str
    MN_VER: str
    N_VER: str

    class Config:
        extra = Extra.forbid


class MakeDataset(BaseModel):
    MDL_ID: str
    MN_VER: str
    N_VER: str
    STYMD: str
    EDYMD: str
    LRNG_DATA_TGT_NCNT: str

    class Config:
        extra = Extra.forbid


class Deploy(BaseModel):
    MDL_ID: str
    MN_VER: str
    N_VER: str
    WDTB_SRVR_NCNT: int

    class Config:
        extra = Extra.forbid


class Predict(BaseModel):
    MDL_ID: str
    MN_VER: str
    N_VER: str
    EVNT_THRU_PATH: list

    class Config:
        extra = Extra.forbid


class AddContainer(BaseModel):
    model_id: str
    version: str
    container_num: int

    class Config:
        extra = Extra.forbid


class RemoveContainer(BaseModel):
    model_id: str
    version: str
    container_num: int

    class Config:
        extra = Extra.forbid


class EndDeploy(BaseModel):
    model_id: str
    version: str

    class Config:
        extra = Extra.forbid


class Train(BaseModel):
    MDL_ID: str
    MN_VER: str
    N_VER: str
    EPOCH: int
    DATA_SPLIT: str
    EARLY_STOP: str
    BATCH_SIZE: int

    class Config:
        extra = Extra.forbid


class StopTrain(BaseModel):
    MDL_ID: str
    MN_VER: str
    N_VER: str

    class Config:
        extra = Extra.forbid


class BasicModelInfo(BaseModel):
    MDL_ID: str
    MN_VER: str
    N_VER: str

    class Config:
        extra = Extra.forbid
