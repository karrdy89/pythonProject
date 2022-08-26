# type and annotation
from typing import Annotated, TypeVar


class InputAnnotation:
    """Input type marker"""


class OutputAnnotation:
    """Output type marker"""


T = TypeVar('T')
Input = Annotated[T, InputAnnotation]

