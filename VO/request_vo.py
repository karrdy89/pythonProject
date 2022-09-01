from pydantic import BaseModel, Extra

class TableInfo(BaseModel):
    table_arr: list[str] = []

    class Config:
        extra = Extra.forbid


class DataInfoContents(BaseModel):
    table_nm: str
    feature_col: list[str] = []
    target_col: str

    class Config:
        extra = Extra.forbid


class DataInfo(BaseModel):
    table_arr: list[DataInfoContents]

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


class CreateTensorboard(BaseModel):
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
