import ray
from ray import actor


@ray.remote
class OnnxServing:
    def __init__(self):
        self._worker = type(self).__name__

    def init(self) -> int:
        return 0

    def _load_model(self):
        pass


# load model
# set attribute
# define input shape and type from metadata
# define output type from metadata(label dict)
# serv with predict method
