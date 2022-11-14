# *********************************************************************************************************************
# Program Name : arg_types
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
from pydantic import BaseModel, Extra


class BasicTableType(BaseModel):
    """
    Define pydantic model to construct inputs of dataset maker
    """
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
