from script.pipeline import Input, Dataset, TrainInfo, PipelineComponent
from script.pipeline import split_ratio
from script.pipeline import base_callbacks, evaluation_callback


@PipelineComponent
def train_test_model(dataset: Input[Dataset], train_info: Input[TrainInfo]):
    import tensorflow as tf

    train_info = train_info
    ds = dataset.data
    ds_size = dataset.length
    train_ratio, validation_ratio, test_ratio = split_ratio(train_info.data_split)
    train_size = int(train_ratio * ds_size)
    validation_size = int(validation_ratio * ds_size)
    train_ds = ds.take(train_size)
    validation_ds = ds.skip(train_size).take(validation_size)
    test_ds = ds.skip(train_size).skip(validation_size)

    train_ds = train_ds.batch(train_info.batch_size)
    validation_ds = validation_ds.batch(train_info.batch_size)
    test_ds = test_ds.batch(train_info.batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    model.compile(optimizer="sgd", loss="mse")
    train_callback = base_callbacks(train_info, monitor="loss")
    test_callback = evaluation_callback(train_info)
    model.fit(train_ds, validation_data=validation_ds, epochs=train_info.epoch, verbose=1, callbacks=train_callback)
    model.evaluate(test_ds, callbacks=test_callback)
    model.save(train_info.save_path)
