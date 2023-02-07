# *********************************************************************************************************************
# Program Name : pipeline
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import yaml
import os
import importlib
import logging
import traceback

import ray

from pipeline import TrainInfo, PipelineComponent, Version
from statics import Actors, TrainStateCode, ROOT_DIR
from pipeline.exceptions import *


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
        self._pipeline_definition_path: str = ROOT_DIR + "/script/pipeline/pipelines.yaml"
        self._logger: ray.actor = ray.get_actor(Actors.LOGGER)
        self._shared_state: ray.actor = ray.get_actor(Actors.GLOBAL_STATE)
        self._pipeline_state: dict[str, str] = {}
        self._components: dict[int, PipelineComponent] = {}
        self._component_result = None
        self._component_idx: int = 0
        self._user_id: str = ''

    def _get_piepline_definition(self) -> dict:
        with open(self._pipeline_definition_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg=str(exc))
                raise exc
            except FileNotFoundError as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg=str(exc))
                raise exc
            except Exception as exc:
                self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg=str(exc))
                raise exc

    def set_pipeline(self, model_id: str, version: str, user_id: str):
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="run: set_pipeline..." + self._name)
        self._name = model_id + ":" + version
        self._user_id = user_id
        try:
            pipeline_list = self._get_piepline_definition()
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="an error occur when parsing yaml file: " + exc.__str__())
            self.on_pipeline_end()
            raise exc
        pipeline_list = pipeline_list.get("pipelines", '')
        if pipeline_list == '':
            self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg="there is no pipeline: " + self._name)
            self.on_pipeline_end()
            raise PipelineNotFoundError()
        sequences = []
        for pipeline in pipeline_list:
            if pipeline.get("name") == model_id:
                sequences = pipeline.get("sequence")
                break
        if len(sequences) == 0:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="there is no sequence in pipeline: " + self._name)
            self.on_pipeline_end()
            raise SequenceNotExistError()
        try:
            for i, seq in enumerate(sequences):
                self._sequence_names.append(seq.get("name"))
                self._pipeline_state[seq.get("name")] = StateCode.WAITING
                task = seq.get("task")
                task_split = task.rsplit('.', 1)
                module = importlib.import_module(task_split[0])
                component = getattr(module, task_split[1])
                self._components[i] = component
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="there is no sequence in pipeline: " + self._name + ": " + exc.__str__())
            self.on_pipeline_end()
            raise SetSequenceError()

    def trigger_pipeline(self, train_info) -> None:
        self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="run: trigger_pipeline..." + self._name)
        if not self._components:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker, msg="there is no component: " + self._name)
            self._shared_state.set_train_status.remote(name=self._name, user_id=self._user_id,
                                                       status_code=TrainStateCode.TRAINING_FAIL)
            self._shared_state.set_error_message.remote(name=self._name, msg="there is no component: " + self._name)
        if self._component_idx >= len(self._components):
            self._component_idx = 0
            self.on_pipeline_end()
            self._logger.log.remote(level=logging.INFO, worker=self._worker, msg="pipeline done: " + self._name)
            self._shared_state.set_train_status.remote(name=self._name, user_id=self._user_id,
                                                       status_code=TrainStateCode.TRAINING_DONE)
            return

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
                        elif v.__name__ == type(Version()).__name__:
                            version = train_info.name.split(':')[-1]
                            args[k] = version
                    self._component_result = component(**args)
        except Exception as e:
            self._pipeline_state[current_task_name] = StateCode.ERROR
            exc_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg=exc_str)
            ray.get(self._shared_state.set_pipeline_result.remote(self._name, self._pipeline_state))
            ray.get(self._shared_state.set_error_message.remote(name=self._name, msg=exc_str))
            ray.get(self._shared_state.set_train_status.remote(name=self._name, user_id=self._user_id,
                                                               status_code=TrainStateCode.TRAINING_FAIL))
            self.on_pipeline_end()
        else:
            self._pipeline_state[current_task_name] = StateCode.DONE
            self._shared_state.set_pipeline_result.remote(self._name, self._pipeline_state)
            self._component_idx += 1
            self.trigger_pipeline(train_info)

    def on_pipeline_end(self) -> None:
        self._shared_state.kill_actor.remote(self._name)

    def kill_process(self) -> int:
        self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                msg="run: kill_process")
        try:
            current_task_name = self._sequence_names[self._component_idx]
            self._pipeline_state[current_task_name] = StateCode.STOP
            ray.get(self._shared_state.set_pipeline_result.remote(self._name, self._pipeline_state))
            ray.get(self._shared_state.set_train_status.remote(self._name, self._user_id, TrainStateCode.TRAINING_FAIL))
        except Exception as exc:
            self._logger.log.remote(level=logging.ERROR, worker=self._worker,
                                    msg="fail: kill_process: " + exc.__str__())
            return -1
        else:
            return 0


class StateCode:
    RUNNING = "RUNNING"
    STOP = "STOP"
    ERROR = "ERROR"
    WAITING = "WAITING"
    DONE = "DONE"
