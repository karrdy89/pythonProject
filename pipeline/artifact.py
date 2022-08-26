from typing import Annotated, TypeVar, Optional


class InputAnnotation:
    """Input type marker"""


class OutputAnnotation:
    """Output type marker"""


T = TypeVar('T')
Input = Annotated[T, InputAnnotation]
Input.__doc__ = """Type generic used to represent an input artifact of type ``T``, where ``T`` is an artifact class."""
Output = Annotated[T, OutputAnnotation]
Output.__doc__ = """Type generic used to represent an output artifact of type ``T``,
where ``T`` is an artifact class."""


class Artifact:
    """
    define base class of Artifact type
    """
    TYPE_NAME = 'pipeline.Artifact'

    def __init__(self, name: Optional[str] = None, metadata: Optional[dict] = None):
        self.name = name or ''
        self.metadata = metadata or {}


class Model(Artifact):
    """
    An artifact representing a machine learning model
    """
    TYPE_NAME = 'pipeline.Model'

    def __init__(self, name: Optional[str] = None, metadata: Optional[dict] = None) -> None:
        super().__init__(name=name, metadata=metadata)

    @property
    def framework(self) -> str:
        return self._get_framework()

    def _get_framework(self) -> str:
        return self.metadata.get('framework', '')

    @framework.setter
    def framework(self, framework: str) -> None:
        self._set_framework(framework)

    def _set_framework(self, framework: str) -> None:
        self.metadata["framework"] = framework


class Dataset(Artifact):
    """
    An artifact representing a machine learning model
    """
    TYPE_NAME = 'pipeline.Model'

    def __init__(self, name: Optional[str] = None, metadata: Optional[dict] = None) -> None:
        super().__init__(name=name, metadata=metadata)

    @property
    def framework(self) -> str:
        return self._get_framework()

    def _get_framework(self) -> str:
        return self.metadata.get('framework', '')

    @framework.setter
    def framework(self, framework: str) -> None:
        self._set_framework(framework)

    def _set_framework(self, framework: str) -> None:
        self.metadata["framework"] = framework


class Url(Artifact):
    """
    An artifact representing an url of data
    """
    TYPE_NAME = 'pipeline.Url'

    def __init__(self, name: Optional[str] = None, metadata: Optional[dict] = None, url: Optional[str] = None) -> None:
        super().__init__(name=name, metadata=metadata)
        self._url = url

    @property
    def url(self) -> str:
        return self._get_url()

    def _get_url(self) -> str:
        return self._url

    @url.setter
    def url(self, url: str) -> None:
        self._set_url(url)

    def _set_url(self, url: str) -> None:
        self._url = url


class Path(Artifact):
    """
    An artifact representing a path of data
    """
    TYPE_NAME = 'pipeline.Path'

    def __init__(self, name: Optional[str] = None, metadata: Optional[dict] = None, path: Optional[str] = None) -> None:
        super().__init__(name=name, metadata=metadata)
        self._path = path

    @property
    def path(self) -> str:
        return self._get_path()

    def _get_path(self) -> str:
        return self._path

    @path.setter
    def path(self, path: str) -> None:
        self._set_path(path)

    def _set_path(self, path: str) -> None:
        self._path = path