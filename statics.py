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
    MODEL_SERVER = "model_serving"
    DATA_MAKER_NBO = "datamaker_nbo"

