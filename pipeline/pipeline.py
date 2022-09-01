# key concept : managed by kubeflow, provide easy GUI(aib), jupyter lab(SDK) interface between kubeflow and distribute
# requires
# 0. set environment and get used kfp
# 1. define input and output type of component
# 2. make container and upload to pipeline
# 3. get pipeline client

# develop first concept once
# define pipeline component
# define input and output


# create input output type class -> plus define metadata (it needs to know model of tensorflow, dataset type)
# base artifact, Dataset Type, Model Type, URL type, Path type, DataFrame Type
# -> how to implement mata? -> 1. make more type includes meta(simple), 2. put meta in output result(better way)
# create component factorizing class(easy to check input and output), make pipeline class connect as order with callback
# train -> log metric, set early stopping, save to defined location
# define pipeline
# ordering and connect input and output in matching
# if matching run pipe and update
# if done update database(after save model, log


import yaml
import os
import importlib
import logging
import traceback

import ray

from pipeline import TrainInfo, PipelineComponent


@ray.remote
class Pipeline:
    def __init__(self):
        self._worker = type(self).__name__
        self._name: str = ''
        self._sequence_names: list[str] = []
        self._pipeline_definition_path: str = os.path.dirname(os.path.abspath(__file__)) + "/pipelines.yaml"
        self._logger: ray.actor = ray.get_actor("logging_service")
        self._shared_state: ray.actor = ray.get_actor("shared_state")
        self._pipeline_state: dict[str, str] = {}
        self._components: dict[int, PipelineComponent] = {}
        self._component_result = None
        self._pipeline_idx: int = 0

    def _get_piepline_definition(self) -> dict:
        with open(self._pipeline_definition_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return {"error": exc}

    def set_pipeline(self, name: str, version: str) -> dict:
        self._name = name + ":" + version
        pipeline_list = self._get_piepline_definition().get("pipelines", '')
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

    def run_pipeline(self, train_info):
        if not self._components:
            return
        if self._pipeline_idx >= len(self._components):
            self._pipeline_idx = 0
            self.on_pipeline_end()
            return
        current_task_name = self._sequence_names[self._pipeline_idx]
        ray.get(self._logger.log.remote(level=logging.INFO, worker=self._worker,
                                        msg="run pipeline step: " + current_task_name))
        self._pipeline_state[current_task_name] = StateCode.RUNNING
        self._shared_state.set_pipeline_result.remote(self._name, self._pipeline_state)
        component = self._components.get(self._pipeline_idx)
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
            self._pipeline_idx += 1
            self.run_pipeline(train_info)

    def on_pipeline_end(self) -> int:
        result = ray.get(self._shared_state.kill_actor.remote(self._name))
        return result


class StateCode:
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    WAITING = "WAITING"
    DONE = "DONE"

#
# shared_state = SharedState.options(name="shared_state").remote()
# logging_service = Logger.options(name="logging_service", max_concurrency=500).remote()
# p = Pipeline()
# p.set_pipeline('test')
# t = TrainInfo()
# t.name = "test"
# t.epoch = 100
# t.data_split = "80:10:10"
# t.early_stop = 'N'
# t.batch_size = '1'
# t.log_path = os.path.dirname(os.path.abspath(__file__)) + '/train_logs/test/101'
# t.save_path = os.path.dirname(os.path.abspath(__file__)) + '/saved_models'
# p.run_pipeline(t)
