from pipeline import Output, Dataset, PipelineComponent


@PipelineComponent
def get_nbo_data() -> Output[Dataset]:
    import pandas as pd
    df = pd.read_csv("dataset/CJ_train.csv")
    out = Dataset()
    out.framework = 'pd'
    out.data = df
    return out
