from pipeline import Input, Output, Dataset, Model, TrainInfo, PipelineComponent
from pipeline.util import split_ratio


@PipelineComponent
def train_test_model(dataset: Input[Dataset], train_info: Input[TrainInfo]):
    import tensorflow as tf
    from tensorflow import keras

    class StateCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(self.params["epochs"])
            self.epoch_step = 0
            progress = (self.epoch_step / self.params["steps"])*100
            progress = str(progress) + "%"
            print(progress)
            # update cur epochs
            # update cur steps

        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            print("End epoch {} of training; got log keys: {}".format(epoch, keys))
            print(logs)
            # update metrics

        # def on_batch_end(self, batch, logs=None):
        #     progress = (self.epoch_step / self.params["steps"])*100
        #     progress = str(progress) + "%"
        #     print(progress)
        #     # update progress

        def on_batch_end(self, batch, logs=None):
            # keys = list(logs.keys())
            # print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
            # print(logs)
            self.epoch_step += 1
            progress = (self.epoch_step / self.params["steps"])*100
            progress = str(progress) + "%"
            print(progress)
    # total epoch num, batch num
    # update : epoch : n/k , progress : j% -> update progress and update epoch to datashare process

    def tensorboard_callback(logdir: str):
        return keras.callbacks.TensorBoard(log_dir=logdir)

    # take train_info and return callback list
    def basic_callbacks(train_inf: TrainInfo, monitor: str) -> list:
        callback_list = []
        tb_cb = tensorboard_callback(train_inf.log_path)
        callback_list.append(tb_cb)
        if train_inf.early_stop == 'Y':
            es_cb = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=10, verbose=1, mode="auto",
                                                  baseline=None, restore_best_weight=True)
            callback_list.append(es_cb)
        callback_list.append(StateCallback())
        return callback_list


    bs_cb = basic_callbacks(train_info, "loss")

    train_info = train_info
    ds = dataset.data
    ds_size = dataset.length
    train_ratio, validation_ratio, test_ratio = split_ratio(train_info.data_split)
    train_size = int(train_ratio * ds_size)
    validation_size = int(validation_ratio * ds_size)
    train_ds = ds.take(train_size)
    validation_ds = ds.skip(train_size).take(validation_size)
    test_ds = ds.skip(train_size).skip(validation_size)

    train_ds = train_ds.batch(1)
    validation_ds = validation_ds.batch(1)
    test_ds = test_ds.batch(1)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    model.compile(optimizer="sgd", loss="mse")
    history = model.fit(train_ds, validation_data=validation_ds, epochs=train_info.epoch, verbose=1, callbacks=bs_cb)
    print(history.history.keys())   # send metric from history  util
    result = model.evaluate(test_ds)
    print(model.metrics_names, result)  # send metric from eval util
    # model.save(train_info.save_path)

    # seperate callback util <-
    # create pipeline and update handle to global state
    # update to global state
    # if pipeline end with 0 -> update pipeline result to database -> if updated kill pipeline process
    # implement nbo
    # test run
