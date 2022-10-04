from pydantic import BaseModel, Extra


class RstCheckTrainProgress(BaseModel):
    MDL_LRNG_ST_CD: str
    CODE: str
    ERROR_MSG: str
    TRAIN_INFO: dict

    class Config:
        extra = Extra.forbid

