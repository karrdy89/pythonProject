# -----
# every module have return
# bind ray tune (1. wrap with func and config, 2. ih tune.Trainable class) -> 1.
# get data, model, config
# define train function, config -> convert to ray tune with pipeline
# is_tune on pipeline param
# run in yaml order
# get module from yaml (validation required), connect, run, save
# pipeline modules must be class
# must know module role (it provides data or model, at least know what model is for train)
# so pipeline information must include it
# how to implement train code? -> check codes and rebuild is unstable. give train option?
# metric, optimizer, callback, etc..
# or just imply train code and call fit -> same as ray tune
# in conclusion pipeline must provide load data(involved preprocess for injection to model), model, train code
# split all is much simple -> then split
# but split is not necessary
# easily merge with kubeflow is the better choice -> dig kubeflow
# vertex Ai = kfp + def + component
# therefore, make sdk binding kfp, distribution
# transport data by argument each pipeline (define arg type of in, out)
# combination of function and decorator

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
import sys
import importlib
import inspect


class Pipeline:
    def __init__(self):
        self._pipeline_definition_path = os.path.dirname(os.path.abspath(__file__)) + "/pipelines.yaml"
        self.progress = {}
        self._components = {}

    def _get_piepline_definition(self) -> dict:
        with open(self._pipeline_definition_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                return {"error": exc}

    def set_pipeline(self, name: str):
        pipeline_list = self._get_piepline_definition().get("pipelines", '')
        # make components, combine sequence if input, output type is same
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
        # combine sequence and check in/out type -> raise error on each comp so, just attach
        # bc first comp has no input(data path is always in the data_loader code(hard coded))
        # maybe recursion <-
        # train here if you want to monitering state or add callback to global state(model_name + state) <-

    def run_pipeline(self):
        for i in range(len(self._components)):
            pass
        comp = self._components.get(0)
        result = comp()
        # check comp.input with before.output
        # if comp.input need mata, inject meta
        # save before state return data, output type. need both?



a = Pipeline()
a.set_pipeline('test')


