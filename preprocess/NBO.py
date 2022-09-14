from pipeline import Input, Output, Dataset, PipelineComponent


@PipelineComponent
def process_test_data(input_data: Input[Dataset]) -> Output[Dataset]:
    import tensorflow as tf
    input_data = input_data.data
    target = input_data.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices((input_data.values, target.values))
    out = Dataset()
    out.framework = 'tf'
    out.length = len(input_data.index)
    out.data = dataset
    return out
