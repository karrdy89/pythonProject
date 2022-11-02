import ray


@ray.remote
class OnnxServingManager:
    def __init__(self):
        self._worker: str = type(self).__name__


# manage actor like tf container
# method : deploy
# method : end_deploy
# method : set cycle
# method : fail back
# method : garbage collect
# method : predict
# method : init
# merge with tf serving? issue is max container num, init process, manage point
# only advantage is performance

