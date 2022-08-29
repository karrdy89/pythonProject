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
from pipeline import TrainInfo


class Pipeline:
    def __init__(self):
        self._pipeline_definition_path = os.path.dirname(os.path.abspath(__file__)) + "/pipelines.yaml"
        self.progress = {}
        self._components = {}
        self._component_result = ComponentResult()
        self._pipeline_idx = 0

    def _get_piepline_definition(self) -> dict:
        with open(self._pipeline_definition_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return {"error": exc}

    def set_pipeline(self, name: str):
        pipeline_list = self._get_piepline_definition().get("pipelines", '')
        sequences = []
        for pipeline in pipeline_list:
            if pipeline.get("name") == name:
                sequences = pipeline.get("sequence")
                break

        for i, seq in enumerate(sequences):
            seq_split = seq.rsplit('.', 1)
            module = importlib.import_module(seq_split[0])
            component = getattr(module, seq_split[1])
            self._components[i] = component

    def run_pipeline(self, train_info):
        if not self._components:
            return 0
        if self._pipeline_idx >= len(self._components):
            self._pipeline_idx = 0
            return 0
        component = self._components.get(self._pipeline_idx)
        inputs = component.input
        outputs = component.output
        if not inputs:
            if not outputs:
                component()
            else:
                self._component_result.output = component()
        else:
            if not outputs:
                if type(TrainInfo()) in inputs:
                    component(self._component_result.output, train_info)    # order checking needed
                else:
                    component(self._component_result.output)
            else:
                if type(TrainInfo()) in inputs:
                    self._component_result.output = component(self._component_result.output, train_info)    # order checking needed
                else:
                    self._component_result.output = component(self._component_result.output)
        self._pipeline_idx += 1
        self.run_pipeline(train_info)


class ComponentResult:
    def __init__(self):
        self.output = None


p = Pipeline()
p.set_pipeline('test')
t = TrainInfo()
t.name = "test"
t.epoch = 100
t.data_split = "80:10:10"
t.early_stop = 'N'
t.batch_size = '1'
p.run_pipeline(t)