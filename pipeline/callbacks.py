import ray
from tensorflow import keras

from pipeline import TrainInfo, TrainResult


class StateCallback(keras.callbacks.Callback):
    def __init__(self, name):
        self._shared_state = ray.get_actor("shared_state")
        self._train_result = TrainResult()
        self.name = name
        self.epoch_step = 0

    def on_train_begin(self, logs=None):
        self._shared_state = ray.get_actor("shared_state")
        self._train_result = TrainResult()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_step = 0
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        self._train_result.set_train_progress(epoch=self.params["epochs"], progress=progress)
        self._shared_state.set_train_result.remote(self.name, self._train_result)

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        print(logs)
        # update metrics

    def on_batch_end(self, batch, logs=None):
        # keys = list(logs.keys())
        # print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
        # print(logs)
        self.epoch_step += 1
        progress = (self.epoch_step / self.params["steps"]) * 100
        progress = str(progress) + "%"
        print(progress)


# total epoch num, batch num
# update : epoch : n/k , progress : j% -> update progress and update epoch to datashare process

# take train_info and return callback list
def basic_callbacks(train_info: TrainInfo, monitor: str) -> list:
    callback_list = []
    tb_cb = keras.callbacks.TensorBoard(log_dir=train_info.log_path)
    callback_list.append(tb_cb)
    if train_info.early_stop == 'Y':
        es_cb = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=10, verbose=1, mode="auto",
                                              baseline=None, restore_best_weight=True)
        callback_list.append(es_cb)
    callback_list.append(StateCallback(train_info.name))
    return callback_list
