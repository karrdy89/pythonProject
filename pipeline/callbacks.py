import ray
from tensorflow import keras

from pipeline import TrainInfo, TrainResult


class BaseCallback(keras.callbacks.Callback):
    def __init__(self, name):
        self._shared_state: ray.actor = ray.get_actor("shared_state")
        self._train_result: TrainResult = TrainResult()
        self.name: str = name
        self.epoch_step: int = 0
        self.epoch: int = 0

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self.epoch_step = 0
        self.epoch += 1
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        self._train_result.set_train_progress(epoch=str(self.epoch)+"/"+str(self.params["epochs"]), progress=progress)
        self._shared_state.set_train_result.remote(self.name, self._train_result)

    def on_epoch_end(self, epoch, logs=None) -> None:
        self._train_result.set_train_result(logs)
        self._shared_state.set_train_result.remote(self.name, self._train_result)

    def on_batch_end(self, batch, logs=None) -> None:
        self.epoch_step += 1
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        self._train_result.set_train_progress(epoch=str(self.epoch)+"/"+str(self.params["epochs"]), progress=progress)
        self._shared_state.set_train_result.remote(self.name, self._train_result)


class EvaluationCallback(keras.callbacks.Callback):
    def __init__(self, name):
        self._shared_state: ray.actor = ray.get_actor("shared_state")
        self._train_result: TrainResult = TrainResult()
        self.name: str = name
        self.epoch_step: int = 0

    def on_test_begin(self, logs=None) -> None:
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        self._train_result.set_test_progress(progress=progress)
        self._shared_state.set_train_result.remote(self.name, self._train_result)

    def on_test_batch_end(self, batch, logs=None):
        self.epoch_step += 1
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        self._train_result.set_test_progress(progress=progress)
        self._shared_state.set_train_result.remote(self.name, self._train_result)

    def on_test_end(self, logs=None):
        self._train_result.set_test_result(logs)
        self._shared_state.set_train_result.remote(self.name, self._train_result)


def base_callbacks(train_info: TrainInfo, monitor: str) -> list:
    callback_list = []
    tb_cb = keras.callbacks.TensorBoard(log_dir=train_info.log_path)
    callback_list.append(tb_cb)
    if train_info.early_stop == 'Y':
        es_cb = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=10, verbose=1, mode="auto",
                                              baseline=None, restore_best_weight=True)
        callback_list.append(es_cb)
    callback_list.append(BaseCallback(train_info.name))
    return callback_list


def evaluation_callback(train_info: TrainInfo) -> list:
    callback_list = [EvaluationCallback(train_info.name)]
    return callback_list

