import ray
from ray import actor
from statics import ROOT_DIR


@ray.remote
class OnnxServing:
    def __init__(self):
        self._worker: str = type(self).__name__
        self._model_path: str = ""

    def init(self, model_id: str, version: str) -> int:
        return 0

    def _load_model(self):
        pass


# load model
# set attribute
# define input shape and type from metadata
# define output type from metadata(label dict)
# serv with predict method
# td
