class PipelineResult:
    def __init__(self):
        self.component_list: list = []
        self.current_component: str = ''


class TrainResult:
    def __init__(self):
        self._train_progress: dict = {}
        self._train_result: dict = {}
        self._test_result: dict = {}

    def set_train_progress(self, progress: dict) -> None:
        self._train_progress = progress

    def get_train_progress(self) -> dict:
        return self._train_progress

    def set_train_result(self, train_result: dict) -> None:
        self._train_result = train_result

    def get_train_result(self) -> dict:
        return self._train_result

    def set_test_result(self, test_result: dict) -> None:
        self._test_result = test_result

    def get_test_result(self) -> dict:
        return self._test_result
