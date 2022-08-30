from pipeline import Input, Dataset, TrainInfo, PipelineComponent
from pipeline.util import split_ratio
from pipeline.callbacks import basic_callbacks


@PipelineComponent
def train_test_model(dataset: Input[Dataset], train_info: Input[TrainInfo]):
    import tensorflow as tf
    from tensorflow import keras


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
    # if pipeline end with 0 -> update pipeline result to database -> if updated kill pipeline process (await for result)
    # implement nbo
    # test run
