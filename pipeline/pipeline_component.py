from pipeline import Artifact
from typing import get_type_hints


class PipelineComponent(object):
    """
    A class that define task of pipeline

    Attributes
    ----------
    func : callable
        function of pipeline task
    output : list
        A name of pipeline. (model_name:version)
    input : dict
        The list of sequence name that defined in pipeline.yaml.

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    _set_output_type() -> None:
        Set output type(return) of given function(pipeline task) in constructor.
    _set_input_types() -> None:
        Set input types of given function(pipeline task) in constructor.
    __call__(self, *args, **kwargs):
        run pipeline task
    """
    def __init__(self, func):
        self.func: callable = func
        self.output: list = []
        self.input: dict = {}
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
        if "return" in types:
            types.pop('return')
        for k, v in types.items():
            if base_artifact == v.__bases__[0]:
                self.input[k] = v

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

