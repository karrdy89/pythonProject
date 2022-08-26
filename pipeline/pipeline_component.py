import sys
from pipeline import Artifact


class PipelineComponent(object):
    def __init__(self, func):
        self._locals = {}
        self.func = func
        self.output = []
        self.input = []
        self._set_output_type()

    def _set_output_type(self) -> None:
        try:
            output_type = self.func.__annotations__["return"]()
        except KeyError:
            pass
        else:
            self.output.append(type(output_type))

    def _set_input_types(self) -> None:
        base_artifact = type(Artifact()).__name__
        for k, v in self._locals.items():
            base_class = None
            for base in v.__class__.__bases__:
                base_class = base.__name__
            if base_class == base_artifact:
                self.input.append(type(v))

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event == "return":
                self._locals = frame.f_locals.copy()
                self._set_input_types()
        sys.setprofile(tracer)
        try:
            res = self.func(*args, **kwargs)
        finally:
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}
