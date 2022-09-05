class TrainResult:
    """
    A class that represents of train progress.

    Attributes
    ----------
    _train_progress : dict[str, str]
        A dictionary of current train progress. ({epoch:current/total, progress:n%})
    _test_progress : dict[str, str]
        A dictionary of current evaluation progress. ({progress:n%})
    _train_result: dict
        A dictionary of current train result. ({metric_1:n, metric_2:k})
    _test_result: dict
        A dictionary of current evaluation result. ({metric_1:n, metric_2:k})

    Methods
    -------
    __init__():
        Constructs all the necessary attributes for the person object.
    set_train_progress(epoch: str, progress: str) -> None:
        Setter of _train_progress.
    get_train_progress() -> dict:
        Getter of _train_progress.
    set_test_progress(progress: str) -> None:
        Setter of _test_progress.
    get_test_progress() -> dict:
        Getter of _test_progress.
    set_train_result(train_result: dict) -> None:
        Setter of _train_result.
    get_train_result() -> dict:
        Getter of _train_result.
    set_test_result(test_result: dict) -> None:
        Setter of _test_result.
    get_test_result() -> dict:
        Getter of _test_result.
    """
    def __init__(self):
        self._train_progress: dict[str, str] = {}
        self._test_progress: dict[str, str] = {}
        self._train_result: dict = {}
        self._test_result: dict = {}

    def set_train_progress(self, epoch: str, progress: str) -> None:
        self._train_progress["epoch"] = epoch
        self._train_progress["progress"] = progress

    def get_train_progress(self) -> dict:
        return self._train_progress

    def set_test_progress(self, progress: str) -> None:
        self._train_progress["progress"] = progress

    def get_test_progress(self) -> dict:
        return self._train_progress

    def set_train_result(self, train_result: dict) -> None:
        self._train_result = train_result

    def get_train_result(self) -> dict:
        return self._train_result

    def set_test_result(self, test_result: dict) -> None:
        self._test_result = test_result

    def get_test_result(self) -> dict:
        return self._test_result
