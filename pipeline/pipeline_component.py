import functools
import inspect
from typing import Callable, List, Optional


def pipeline_component(func: Optional[Callable] = None, ):  # decorator class connect input and output
    pass

# get input output class

def a(s: str, b: int) -> int:
    pass

sig = inspect.signature(a)
sig.parameters

for param in sig.parameters.values():
    print(param.name)
    print(param.annotation)

a.__annotations__["return"]

# create input output type class -> plus define metadata (it needs to know model of tensorflow, dataset type)
# base artifact, Dataset Type, Model Type, URL type, Path type, DataFrame Type
# -> how to implement mata? -> 1. make more type includes meta(simple), 2. put meta in output result(better way)
# create component factorizing class(easy to check input and output), make pipeline class connect as order with callback
# train -> log metric, set early stopping, save to defined location
# define pipeline
# ordering and connect input and output in matching
# if matching run pipe and update
# if done update database(after save model, log

