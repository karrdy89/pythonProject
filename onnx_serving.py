import ray
from ray import actor


@ray.remote
class OnnxServing:
    def __init__(self):
        self._worker = type(self).__name__
