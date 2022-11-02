import ray


@ray.remote
class OnnxServingManager:
    def __init__(self):
        self._worker: str = type(self).__name__
