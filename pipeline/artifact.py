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

    def __init__(self, name: Optional[str] = None, 
                 data: Optional[object | str] = None, metadata: Optional[dict] = None):
        self.name = name or ''
        self.data = data or None
        self.metadata = metadata or {}


class Model(Artifact):
    """
    An artifact representing a machine learning model
    """
    TYPE_NAME = 'pipeline.Model'

    def __init__(self, name: Optional[str] = None, 
                 data: Optional[object | str] = None, metadata: Optional[dict] = None):
        super().__init__(name=name, data=data, metadata=metadata)

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
        
    @property
    def model(self) -> object:
        return self._get_model()

    def _get_model(self) -> object:
        return self.model

    @model.setter
    def model(self, model: object) -> None:
        self._set_model(model)

    def _set_model(self, model: object) -> None:
        self.model = model


class Dataset(Artifact):
    """
    An artifact representing a machine learning model
    """
    TYPE_NAME = 'pipeline.Model'

    def __init__(self, name: Optional[str] = None, 
                 data: Optional[object | str] = None, metadata: Optional[dict] = None):
        super().__init__(name=name, data=data, metadata=metadata)

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

    def __init__(self, name: Optional[str] = None, 
                 data: Optional[object | str] = None, metadata: Optional[dict] = None):
        super().__init__(name=name, data=data, metadata=metadata)

    @property
    def url(self) -> str:
        return self._get_url()

    def _get_url(self) -> str:
        return self.data

    @url.setter
    def url(self, url: str) -> None:
        self._set_url(url)

    def _set_url(self, url: str) -> None:
        self.data = url


class Path(Artifact):
    """
    An artifact representing a path of data
    """
    TYPE_NAME = 'pipeline.Path'

    def __init__(self, name: Optional[str] = None, 
                 data: Optional[object | str] = None, metadata: Optional[dict] = None):
        super().__init__(name=name, data=data, metadata=metadata)

    @property
    def path(self) -> str:
        return self._get_path()

    def _get_path(self) -> str:
        return self.data

    @path.setter
    def path(self, path: str) -> None:
        self._set_path(path)

    def _set_path(self, path: str) -> None:
        self.data = path


class TrainInfo(Artifact):
    """
    An artifact representing a path of data
    """
    TYPE_NAME = 'pipeline.train_info'

    def __init__(self, name: Optional[str] = None,
                 data: Optional[object | str] = None, metadata: Optional[dict] = None):
        super().__init__(name=name, data=data, metadata=metadata)

    @property
    def epoch(self) -> int:
        return self._get_epoch()

    def _get_epoch(self) -> int:
        return self.metadata.get("epoch", 0)

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        self._set_epoch(epoch)

    def _set_epoch(self, epoch: int) -> None:
        self.metadata["epoch"] = epoch

    @property
    def batch_size(self) -> int:
        return self._get_batch_size()

    def _get_batch_size(self) -> int:
        return self.metadata.get("batch_size", 0)

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._set_batch_size(batch_size)

    def _set_batch_size(self, batch_size: int) -> None:
        self.metadata["batch_size"] = batch_size

    @property
    def data_split(self) -> str:
        return self._get_data_split()

    def _get_data_split(self) -> str:
        return self.metadata.get("data_split", '')

    @data_split.setter
    def data_split(self, data_split: str) -> None:
        self._set_data_split(data_split)

    def _set_data_split(self, data_split: str) -> None:
        self.metadata["data_split"] = data_split

    @property
    def early_stop(self) -> str:
        return self._get_early_stop()

    def _get_early_stop(self) -> str:
        return self.metadata.get("early_stop", '')

    @early_stop.setter
    def early_stop(self, early_stop: str) -> None:
        self._set_early_stop(early_stop)

    def _set_early_stop(self, early_stop: str) -> None:
        self.metadata["early_stop"] = early_stop
