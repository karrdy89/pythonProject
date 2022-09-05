import logging
from collections import OrderedDict

import ray
from ray import actor

from pipeline import TrainResult


PIPELINE_MAX = 1


@ray.remote
class SharedState:
    def __init__(self):
        self._worker = type(self).__name__
        self._logger: actor = None
        self._actors: OrderedDict[str, actor] = OrderedDict()
        self._pipline_result: OrderedDict[str, dict] = OrderedDict()
        self._train_result: OrderedDict[str, TrainResult] = OrderedDict()
        self._pipeline_pool: OrderedDict[str, actor] = OrderedDict()

    def init(self):
        self._logger: actor = ray.get_actor("logging_service")

    def set_actor(self, name: str, act: actor):
        self._actors[name] = act
        if len(self._actors) >= PIPELINE_MAX:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="max piepline exceeded")
            return -1
        if len(self._pipline_result) >= PIPELINE_MAX:
            self._pipline_result.popitem(last=False)
            self._train_result.popitem(last=False)

    def is_actor_exist(self, name) -> bool:
        if name in self._actors:
            return True
        return False

    def delete_actor(self, name: str) -> None:
        if name in self._actors:
            del self._actors[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="actor not exist: " + name)

    def kill_actor(self, name: str) -> int:
        if name in self._actors:
            act = self._actors[name]
            ray.kill(act)
            self.delete_actor(name)
            return 0
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="actor not exist: " + name)
            return -1

    def set_pipeline_result(self, name: str, pipe_result: dict) -> None:
        self._pipline_result[name] = pipe_result

    def delete_pipeline_result(self, name: str) -> None:
        if name in self._pipline_result:
            del self._pipline_result[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="pipeline not exist: " + name)

    def get_pipeline_result(self, name: str) -> dict:
        if name in self._pipline_result:
            pipeline = self._pipline_result[name]
            return {"component_list": pipeline}
        else:
            return {}

    def set_train_result(self, name: str, train_result: TrainResult) -> None:
        self._train_result[name] = train_result

    def get_train_result(self, name: str) -> dict:
        if name in self._train_result:
            train_result = self._train_result[name]
            return {"progress": train_result.get_train_progress(), "train_result": train_result.get_train_result(),
                    "test_result": train_result.get_test_result()}
        else:
            return {}

    def delete_train_result(self, name: str) -> None:
        if name in self._train_result:
            del self._train_result[name]
        else:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="pipeline not exist: " + name)
