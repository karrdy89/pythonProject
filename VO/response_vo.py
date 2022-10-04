from pydantic import BaseModel, Extra


class BasicModelInfo(BaseModel):
    RESULT: dict

    class Config:
        extra = Extra.forbid
