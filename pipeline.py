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


import yaml
import os
import sys
import importlib

p_yaml = None
with open(os.path.dirname(os.path.abspath(__file__)) + "/pipelines.yaml", 'r') as stream:
    try:
        p_yaml = yaml.safe_load(stream)
        print(p_yaml)
    except yaml.YAMLError as exc:
        print(exc)

data_loader = getattr(importlib.import_module("data_loader.test"), "Test")
module = getattr(importlib.import_module("models.test"), "Test")
ins = module()
m = ins.pipe()
m.summary()
