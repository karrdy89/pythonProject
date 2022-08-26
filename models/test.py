from pipeline import Output, Model, PipelineComponent


@PipelineComponent
def test_model() -> Output[Model]:
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    model.compile(optimizer="sgd", loss="mse")
    out = Model()
    out.framework = 'tf'
    out.data = model
    return out