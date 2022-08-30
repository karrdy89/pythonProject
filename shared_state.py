import ray
from ray import actor

from pipeline import PipelineResult, TrainResult


@ray.remote
class SharedState:
    def __init__(self):
        self._actors: dict[str, actor] = {} # if not work change to actor name
        self._pipline_result: dict[str, PipelineResult] = {}
        self._train_result: dict[str, TrainResult] = {}

    def set_actor(self, name: str, act: actor) -> None:
        self._actors[name] = act

    def delete_actor(self, name: str) -> None:
        if name in self._actors:
            del self._actors[name]
        else:
            print(name+" actor not exist")

    def kill_actor(self, name: str) -> None:
        if name in self._actors:
            act = self._actors["name"]
            ray.kill(act)
            self.delete_actor(name)
        else:
            print(name+" actor not exist")

    def set_pipeline_result(self, name: str, pipe_result: PipelineResult) -> None:
        self._pipline_result[name] = pipe_result

    def delete_pipeline_result(self, name: str) -> None:
        if name in self._pipline_result:
            del self._pipline_result[name]
        else:
            print(name+" pipeline not exist")

    def get_pipeline_result(self, name: str) -> dict:
        if name in self._pipline_result:
            pipeline = self._pipline_result[name]
            return {"component_list": pipeline.component_list, "current_process": pipeline.current_component}
        else:
            return {}

    def set_train_result(self, name: str, train_result: TrainResult) -> None:
        self._train_result[name] = train_result

    def get_train_result(self, name: str) -> dict:
        if name in self._train_result:
            pass

    def delete_train_result(self, name: str) -> None:
        if name in self._train_result:
            del self._train_result[name]
        else:
            print(name+" pipeline not exist")

# handle of actor -> {'name': handle}
# pipeline result -> {'name': pipeline result}
# pipeline result class -> {'progress_list':pipeline list, 'current_process':process name}
# train result -> {'name': train result}
# train result class -> train_progress {'epoch':n/k, 'rate':%}, train result {'metric_1':0.05, 'metric_2':0.01}, test result {'metric_1':0.05, 'metric_2':0.01}