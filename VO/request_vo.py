from pydantic import BaseModel, Extra


class CheckTrainProgress(BaseModel):
    MDL_ID: str
    MDL_NM: str
    MN_VER: int
    N_VER: int

    class Config:
        extra = Extra.forbid


class MakeDataset(BaseModel):
    MDL_ID: str
    MDL_NM: str
    MN_VER: int
    N_VER: int
    STYMD: str
    EDYMD: str
    LRNG_DATA_TGT_NCNT: str

    class Config:
        extra = Extra.forbid


class Deploy(BaseModel):
    model_id: str
    version: str
    container_num: int

    class Config:
        extra = Extra.forbid


class Predict(BaseModel):
    model_id: str
    version: str
    feature: dict

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
    model_id: str
    version: str
    epoch: int
    data_split: str
    early_stop: str
    batch_size: int

    class Config:
        extra = Extra.forbid
