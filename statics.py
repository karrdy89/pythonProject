# *********************************************************************************************************************
# Program Name : statics
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class ErrorCode:
    NOT_FOUND = 4
    DUPLICATED_REQUEST = 5
    FILE_IO = 6
    EXCEED_LIMITATION = 7
    HTTP_REQUEST = 8
    DOCKER = 9


class Actors:
    LOGGER = "logging_service"
    GLOBAL_STATE = "shared_state"
    SERVER = "API_service"
    SERVER_OP = "API_service_op"
    SERVING_MANAGER = "serving_manager"
    DATA_MAKER_NBO = "datamaker_nbo"


class TrainStateCode:
    MAKING_DATASET = "11"
    MAKING_DATASET_DONE = "12"
    MAKING_DATASET_FAIL = "13"
    TRAINING = "21"
    TRAINING_DONE = "22"
    TRAINING_FAIL = "23"


class ModelType:
    Tensorflow = "TF"
    ONNX = "ONNX"
