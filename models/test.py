import tensorflow as tf
from pipeline import Output, Model, PipelineComponent


@PipelineComponent
def test_model() -> Output[Model]:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    model.compile(optimizer="sgd", loss="mse")
    out = Model()
    out.framework = 'tf'
    out.data = model
    return out


# model.fit(xs, ys, epochs=1000)

# model.save(ROOT_DIR+'/saved_models/test/101')
