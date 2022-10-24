from pipeline import Input, Output, Dataset, PipelineComponent


@PipelineComponent
def process_nbo_data(input_data: Input[Dataset]) -> Output[Dataset]:
    import pandas as pd
    input_data = input_data.data
    labels = input_data['Target'].unique()
    labels = labels.tolist()
    samples = {}
    for i in labels:
        samples[i] = input_data["Target"].value_counts()[i]
    min_sample = min(samples, key=samples.get)
    sep_frames = []
    for label in labels:
        if label is not min_sample:
            downed = input_data[input_data['Target'] == label]
            downed = downed.sample(n=samples[min_sample].item(), random_state=0)
            sep_frames.append(downed)
    df = pd.concat(sep_frames, axis=0)
    df.fillna('', inplace=True)
    out = Dataset()
    out.framework = 'pd'
    out.length = len(input_data.index)
    out.data = df
    return out
