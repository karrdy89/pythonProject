from pydantic import BaseModel, Extra


class TableInfoVO(BaseModel):
    table_arr: list[str] = []

    class Config:
        extra = Extra.forbid


class DataInfoContents(BaseModel):
    table_nm: str
    feature_col: list[str] = []
    target_col: str

    class Config:
        extra = Extra.forbid


class DataInfoVO(BaseModel):
    table_arr: list[DataInfoContents]

    class Config:
        extra = Extra.forbid


class DeployVO(BaseModel):
    model_id: str
    model_version: str
    container_num: int

    class Config:
        extra = Extra.forbid


class PredictVO(BaseModel):
    model_name: str
    feature: list

    class Config:
        extra = Extra.forbid