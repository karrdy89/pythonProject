from pydantic import BaseModel, Extra


class BasicTableType(BaseModel):
    actor_name: str
    dataset_name: str
    version: str
    actor_handle: object
    query_name: str
    labels: list
    key_index: int
    feature_index: list
    num_data_limit: int
    start_dtm: str
    end_dtm: str

    class Config:
        extra = Extra.forbid
