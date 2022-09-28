import ray
from tensorflow import keras

from pipeline import TrainInfo
from statics import Actors


class BaseCallback(keras.callbacks.Callback):
    """
    A callback class to monitor training progress.

    Attributes
    ----------
    _shared_state : actor
        an actor handle of global data store.
    name : str
        a name of pipeline.
    epoch_step : int
        a batch of each epoch.
    epoch : int
        current epoch of training.

    Methods
    -------
    __init__(name: str):
        Constructs all the necessary attributes.
    on_epoch_begin(epoch, logs=None) -> None:
        update training progress to global data store when epoch begin.
    on_epoch_end(epoch, logs=None) -> None:
        update training progress to global data store when epoch end.
    on_batch_end(batch, logs=None) -> None:
        update training progress to global data store when batch end.
    """
    def __init__(self, name):
        self._shared_state: ray.actor = ray.get_actor(Actors.GLOBAL_STATE)
        self.name: str = name
        self.epoch_step: int = 0
        self.epoch: int = 0

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self.epoch_step = 0
        self.epoch += 1
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        # self._train_result.set_train_progress(epoch=str(self.epoch)+"/"+str(self.params["epochs"]), progress=progress)
        self._shared_state.set_train_progress.remote(name=self.name,
                                                     epoch=str(self.epoch)+"/"+str(self.params["epochs"]),
                                                     progress=progress)
        self._shared_state.set_train_result.remote(name=self.name, train_result=logs)

    def on_epoch_end(self, epoch, logs=None) -> None:
        self._shared_state.set_train_result.remote(name=self.name, train_result=logs)

    def on_batch_end(self, batch, logs=None) -> None:
        self.epoch_step += 1
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        self._shared_state.set_train_progress.remote(name=self.name,
                                                     epoch=str(self.epoch)+"/"+str(self.params["epochs"]),
                                                     progress=progress)
        self._shared_state.set_train_result.remote(name=self.name, train_result=logs)


class EvaluationCallback(keras.callbacks.Callback):
    """
    A callback class to monitor evaluation progress.

    Attributes
    ----------
    _shared_state : actor
        an actor handle of global data store.
    _train_result : TrainResult
        a current result of training.
    name : str
        a name of pipeline.
    epoch_step : int
        a batch of each epoch.

    Methods
    -------
    __init__(name: str):
        Constructs all the necessary attributes.
    on_test_begin(logs=None) -> None:
        update training progress to global data store when evaluation begin.
    on_test_batch_end(batch, logs=None) -> None:
        update training progress to global data store when batch end.
    on_test_end(logs=None) -> None:
        update training progress to global data store when evaluation end.
    """
    def __init__(self, name):
        self._shared_state: ray.actor = ray.get_actor("shared_state")
        self.name: str = name
        self.epoch_step: int = 0

    def on_test_begin(self, logs=None) -> None:
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        self._shared_state.set_test_progress.remote(name=self.name, progress=progress)
        self._shared_state.set_test_result.remote(self.name, logs)

    def on_test_batch_end(self, batch, logs=None) -> None:
        self.epoch_step += 1
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        self._shared_state.set_test_progress.remote(name=self.name, progress=progress)
        self._shared_state.set_test_result.remote(self.name, logs)

    def on_test_end(self, logs=None) -> None:
        self._shared_state.set_test_result.remote(self.name, logs)


def base_callbacks(train_info: TrainInfo, monitor: str) -> list:
    """
    Provides necessary callbacks for train(progress monitoring, early stopping, tensorboard log callback)

    Parameters
    ----------
    train_info : TrainInfo
        training options
    monitor : str
        value to monitor for early stopping
    """
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
    """
    Provides necessary callbacks for evaluation(progress monitoring)

    Parameters
    ----------
    train_info : TrainInfo
        training options
    """
    callback_list = [EvaluationCallback(train_info.name)]
    return callback_list

