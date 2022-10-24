from pipeline import Version, Input, Output, Dataset, PipelineComponent
from statics import ROOT_DIR


@PipelineComponent
def get_nbo_data(version: Input[Version]) -> Output[Dataset]:
    mn_ver = version.split('.')[-1]
    dataset_path = ROOT_DIR + "/dataset/MDL0000001/" + mn_ver + "/"
    print(dataset_path)
    import pandas as pd
    df = pd.read_csv("dataset/CJ_train.csv")
    out = Dataset()
    out.framework = 'pd'
    out.data = df
    return out
