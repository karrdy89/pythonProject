# *********************************************************************************************************************
# Program Name : callbacks
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
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

    def __init__(self, name: str, b_steps: int | None = 0, t_epoch: int | None = 1, c_epoch: int | None = 0,
                 t_steps: int | None = None, t_file_count: int | None = 1, c_file_count: int | None = 1):
        self._shared_state: ray.actor = ray.get_actor(Actors.GLOBAL_STATE)
        self.name: str = name
        self.epoch_step: int = 0
        self.cur_steps: int = b_steps
        self.t_steps = t_steps
        self.epoch: int = c_epoch
        self.t_epoch = t_epoch
        self.total: int = 0
        self.t_file_count = t_file_count
        self.c_file_count = c_file_count

    def on_train_begin(self, logs=None):
        if self.t_steps is not None:
            self.total = self.t_epoch * self.t_steps
        else:
            # c_total = self.params["steps"] * self.t_epoch
            # c_total = self.params["steps"] * self.dataset_num
            if self.t_file_count < 2:
                self.total = self.params["steps"]*self.t_epoch + self.params["steps"] * (self.t_file_count - self.c_file_count) * self.t_epoch
            else:
                self.total = self.cur_steps + self.params["steps"] * self.t_epoch + self.params["steps"] * (
                            self.t_file_count - self.c_file_count) * self.t_epoch

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self.epoch_step = 0
        # self.epoch += 1
        progress = (self.epoch_step / self.params["steps"])
        total_progress = self.cur_steps / self.total
        self._shared_state.set_train_progress.remote(name=self.name,
                                                     epoch=str(self.epoch) + "/" + str(self.t_epoch),
                                                     progress=progress,
                                                     total_progress=total_progress)
        self._shared_state.set_train_result.remote(name=self.name, train_result=logs)

    def on_epoch_end(self, epoch, logs=None) -> None:
        self._shared_state.set_train_result.remote(name=self.name, train_result=logs)

    def on_batch_end(self, batch, logs=None) -> None:
        self.epoch_step += 1
        self.cur_steps += 1
        progress = (self.epoch_step / self.params["steps"])
        total_progress = self.cur_steps / self.total
        self._shared_state.set_train_progress.remote(name=self.name,
                                                     epoch=str(self.epoch) + "/" + str(self.t_epoch),
                                                     progress=progress,
                                                     total_progress=total_progress)
        self._shared_state.set_train_result.remote(name=self.name, train_result=logs)


class EvaluationCallback(keras.callbacks.Callback):
    """
    A callback class to monitor evaluation progress.

    Attributes
    ----------
    _shared_state : actor
        an actor handle of global data store.
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
        progress = (self.epoch_step / self.params["steps"])
        self._shared_state.set_test_progress.remote(name=self.name, progress=progress)
        self._shared_state.set_test_result.remote(self.name, logs)

    def on_test_batch_end(self, batch, logs=None) -> None:
        self.epoch_step += 1
        progress = (self.epoch_step / self.params["steps"])
        self._shared_state.set_test_progress.remote(name=self.name, progress=progress)
        self._shared_state.set_test_result.remote(self.name, logs)

    def on_test_end(self, logs=None) -> None:
        self._shared_state.set_test_result.remote(self.name, logs)


def base_callbacks(train_info: TrainInfo, b_steps: int, t_epoch: int, c_epoch: int, t_steps: int | None
                   ,t_file_count: int ,c_file_count: int) -> list:
    """
    Provides necessary callbacks for train(progress monitoring, early stopping, tensorboard log callback)

    Parameters
    ----------
    train_info : TrainInfo
        training options
    monitor : str
        value to monitor for early stopping
    dataset_num : int
    cur_dataset_num : int
    """
    callback_list = []
    tb_cb = keras.callbacks.TensorBoard(log_dir=train_info.log_path)
    callback_list.append(tb_cb)
    callback_list.append(BaseCallback(train_info.name, b_steps, t_epoch, c_epoch, t_steps, t_file_count, c_file_count))
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
