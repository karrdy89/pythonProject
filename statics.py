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
    TF_SERVING = "tf_serving"
    ONNX_SERVING = "onnx_serving"
    DATA_MAKER_NBO = "datamaker_nbo"


class TrainStateCode:
    MAKING_DATASET = "11"
    MAKING_DATASET_DONE = "12"
    MAKING_DATASET_FAIL = "13"
    TRAINING = "21"
    TRAINING_DONE = "22"
    TRAINING_FAIL = "23"


class ModelType:
    Tensorflow = 0
    ONNX = 1


class ModelInfo:
    def __init__(self, model_name: str, model_type: int):
        self.model_name = model_name
        self.model_type = model_type


class BuiltinModels:
    MDL0000001 = ModelInfo(model_name="NBO", model_type=ModelType.Tensorflow)
