import yaml
import os
import importlib
import logging
import traceback

import ray

from pipeline import TrainInfo, PipelineComponent


@ray.remote
class Pipeline:
    """
    A ray actor class that carry out pipeline task

    Attributes
    ----------
    _worker : str
        Class name of instance.
    _name : str
        A name of pipeline. (model_name:version)
    _sequence_names : list[str]
        The list of sequence name that defined in pipeline.yaml.
    _pipeline_definition_path : str
        The path to pipeline definition file.
    _logger : actor
        The actor handel of global logger.
    _shared_state : actor
        The actor handel of global data store.
    _pipeline_state : dict[str, str]
        Progress of current pipeline. ({sequence_name_1:state_code, sequence_name_2:state_code, ...})
    _components : dict[int, PipelineComponent]
        Dictionary of pipeline components. ({task_idx:component})
    _component_result: Any
        The return of prior component.
    _component_idx: int
        An index of current pipeline component.

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    _get_piepline_definition() -> dict:
        Create dictionary from pipeline.yaml.
    run_pipeline(name: str, version: str, train_info: TrainInfo) -> dict:
        Set pipline attributes and run pipeline.
    trigger_pipeline(train_info) -> int:
        Run each component of piepline
    on_pipeline_end() -> None:
        Ask shared_state actor to kill this pipeline
    """
    def __init__(self):
        self._worker = type(self).__name__
        self._name: str = ''
        self._sequence_names: list[str] = []
        self._pipeline_definition_path: str = os.path.dirname(os.path.abspath(__file__)) + "/pipeline/pipelines.yaml"
        self._logger: ray.actor = ray.get_actor("logging_service")
        self._shared_state: ray.actor = ray.get_actor("shared_state")
        self._pipeline_state: dict[str, str] = {}
        self._components: dict[int, PipelineComponent] = {}
        self._component_result = None
        self._component_idx: int = 0

    def _get_piepline_definition(self) -> dict:
        with open(self._pipeline_definition_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg=str(exc))
                return {"error": exc}

    def run_pipeline(self, name: str, version: str, train_info: TrainInfo) -> dict:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="set pipeline..." + self._name)
        self._name = name + ":" + version
        pipeline_list = self._get_piepline_definition()
        if "error" in pipeline_list:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when parsing yaml file")
            return {"error": "an error occur when parsing yaml file"}
        pipeline_list.get("pipelines", '')
        if pipeline_list == '':
            self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg="there is no pipeline: " + self._name)
            return {"result": "there is no pipeline: " + name}
        sequences = []
        for pipeline in pipeline_list:
            if pipeline.get("name") == name:
                sequences = pipeline.get("sequence")
                break
        for i, seq in enumerate(sequences):
            self._sequence_names.append(seq.get("name"))
            self._pipeline_state[seq.get("name")] = StateCode.WAITING
            task = seq.get("task")
            task_split = task.rsplit('.', 1)
            module = importlib.import_module(task_split[0])
            component = getattr(module, task_split[1])
            self._components[i] = component
        pipeline_result = self.trigger_pipeline(train_info=train_info)
        if pipeline_result == 0:
            return {"result": "pipeline finished successfully : " + name}
        else:
            return {"result": "pipeline fail : " + name}

    def trigger_pipeline(self, train_info) -> int:
        if not self._components:
            self._logger.log.remote(level=logging.WARN, worker=self._worker, msg="there is no component: " + self._name)
            return -1
        if self._component_idx >= len(self._components):
            self._component_idx = 0
            self.on_pipeline_end()
            self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="end pipeline..." + self._name)
            return 0
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="run pipeline..." + self._name)
        current_task_name = self._sequence_names[self._component_idx]
        ray.get(self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="run pipeline step: " + current_task_name))
        self._pipeline_state[current_task_name] = StateCode.RUNNING
        self._shared_state.set_pipeline_result.remote(self._name, self._pipeline_state)
        component = self._components.get(self._component_idx)
        inputs = component.input
        outputs = component.output
        try:
            if not inputs:
                if not outputs:
                    component()
                else:
                    self._component_result = component()
            else:
                args = {}
                if not outputs:
                    for k, v in inputs.items():
                        if v.__name__ == type(self._component_result).__name__:
                            args[k] = self._component_result
                        elif v.__name__ == type(TrainInfo()).__name__:
                            args[k] = train_info
                    component(**args)
                else:
                    for k, v in inputs.items():
                        if v.__name__ == type(self._component_result).__name__:
                            args[k] = self._component_result
                        elif v.__name__ == type(TrainInfo()).__name__:
                            args[k] = train_info
                    self._component_result = component(**args)
        except Exception as e:
            self._pipeline_state[current_task_name] = StateCode.ERROR
            exc_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg=exc_str)
            self._shared_state.set_pipeline_result.remote(self._name, self._pipeline_state)
            return -1
        else:
            self._pipeline_state[current_task_name] = StateCode.DONE
            self._shared_state.set_pipeline_result.remote(self._name, self._pipeline_state)
            self._component_idx += 1
            self.trigger_pipeline(train_info)

    def on_pipeline_end(self) -> None:
        ray.get(self._shared_state.kill_actor.remote(self._name))


class StateCode:
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    WAITING = "WAITING"
    DONE = "DONE"
