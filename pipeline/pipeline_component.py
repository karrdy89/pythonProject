import sys
from pipeline import Artifact
from typing import get_type_hints


class PipelineComponent(object):
    def __init__(self, func):
        self.func = func
        self.output: list = []
        self.input: dict = {} # key val, for assemble input with argument(order check)
        self._set_output_type()
        self._set_input_types()

    def _set_output_type(self) -> None:
        try:
            output_type = self.func.__annotations__["return"]()
        except KeyError:
            pass
        else:
            self.output.append(type(output_type))

    def _set_input_types(self) -> None:
        base_artifact = type(Artifact())
        types = get_type_hints(self.func)
        print(types)
        if "return" in types:
            types.pop('return')
        for k, v in types.items():
            if base_artifact == v.__bases__[0]:
                self.input[k] = v
                # self.input.append(v)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

