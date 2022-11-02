import ray
from ray import actor


@ray.remote
class OnnxServing:
    def __init__(self):
        self._worker = type(self).__name__

# load model
# set attribute
# define input shape and type
# define output type
# serv with predict method
