import functools
import inspect
from typing import Callable, List, Optional, get_type_hints
from pipeline import Input, Output, Dataset, Model


def pipeline_component(func: Optional[Callable] = None, ):  # decorator class connect input and output
    pass

# get input output class

# def a(s: str, b: int) -> int:
#     pass
#
# sig = inspect.signature(a)
# sig.parameters
#
# for param in sig.parameters.values():
#     print(param.name)
#     print(param.annotation)
#
# a.__annotations__["return"]

# create input output type class -> plus define metadata (it needs to know model of tensorflow, dataset type)
# base artifact, Dataset Type, Model Type, URL type, Path type, DataFrame Type
# -> how to implement mata? -> 1. make more type includes meta(simple), 2. put meta in output result(better way)
# create component factorizing class(easy to check input and output), make pipeline class connect as order with callback
# train -> log metric, set early stopping, save to defined location
# define pipeline
# ordering and connect input and output in matching
# if matching run pipe and update
# if done update database(after save model, log


def aa(model: Input[Model], dataset: Output[Dataset]) -> Output[Model]:
    model.framework = 'tf'
    print(model.framework)
    pass

aa(Model(), Dataset())
sig = inspect.signature(aa)
sig.parameters

for param in sig.parameters.values():
    print(param.name)
    print(type(param.annotation))
    print(param.annotation)

aa.__annotations__["return"]

print(type(aa.__annotations__["return"]))




import sys

class persistent_locals(object):    # extract typing anotation is all i need
    def __init__(self, func):
        self._locals = {}
        self.func = func
        print(func.__annotations__["return"])

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event=='return':
                self._locals = frame.f_locals.copy()

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals

@persistent_locals
def aa(model: Input[Model], dataset: Output[Dataset]) -> Output[Model]:
    model.framework = 'tf'
    print(model.framework)
    pass

aa(Model(), Dataset())
print(aa.locals)