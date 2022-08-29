from pipeline import Input, Output, Dataset, PipelineComponent


@PipelineComponent
def process_test_data(input_data: Input[Dataset]) -> Output[Dataset]:
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    target = input_data.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices(input_data.values, target.values)
    out = Dataset()
    Dataset.framework = 'tf'
    Dataset.data = dataset
    return out

