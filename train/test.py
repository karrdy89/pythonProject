from pipeline import Input, Dataset, TrainInfo, PipelineComponent


@PipelineComponent
def train_test_model(dataset: Input[Dataset], train_info: Input[TrainInfo]):
    import tensorflow as tf

    train_info = train_info
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    ds = dataset.data
    model.compile(optimizer="sgd", loss="mse")
    for feat, targ in ds.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))

    ds = ds.batch(1)

    model.fit(ds, epochs=10)


    #save model
    #save log
    #early stopping
    #update to grobal state
    #need train metadata -> name and train option
    # train here if you want to monitering state or add callback to global state(model_name + state)