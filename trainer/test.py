from pipeline import Input, Dataset, Model, PipelineComponent


@PipelineComponent
def train_test_model(input_data: Input[Dataset], input_model: Input[Model]):
    import tensorflow as tf
    import pandas as pd
    df = input_data.data
    print(type(input_data.data))
    df.__class__ = type(pd.DataFrame())
    model = input_model.model

    # dataframe to dataset
    # model.fit(xs, ys, epochs=1000)
    # model.save(ROOT_DIR+'/saved_models/test/101')
