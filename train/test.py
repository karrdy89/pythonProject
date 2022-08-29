from pipeline import Input, Dataset, TrainInfo, PipelineComponent
from pipeline.util import split_ratio


@PipelineComponent
def train_test_model(dataset: Input[Dataset], train_info: Input[TrainInfo]):
    import tensorflow as tf
    from tensorflow import keras

    class TestCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            print(11)

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

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    model.compile(optimizer="sgd", loss="mse")
    model.fit(train_ds, validation_data=validation_ds, epochs=train_info.epoch, verbose=1, callbacks=[TestCallback()])

    #save model
    #save log
    #early stopping
    #update to grobal state
    #need train metadata -> name and train option
    # train here if you want to monitering state or add callback to global state(model_name + state)