from pipeline import Input, Dataset, PipelineComponent


@PipelineComponent
def train_test_model(input_data: Input[Dataset]):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    model.compile(optimizer="sgd", loss="mse")
    model.fit(input_data, epochs=10)

    #save model
    #save log
    #early stopping
    #update to grobal state
    #need train metadata -> name and train option