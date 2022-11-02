from pipeline import Input, TrainInfo, PipelineComponent
from statics import ROOT_DIR


@PipelineComponent
def train_FD_model(train_info: Input[TrainInfo]):
    sp_model_info = train_info.name.split(':')
    model_name = sp_model_info[0]
    mn_version = sp_model_info[-1].split('.')[0]
    # base_dataset_path = ROOT_DIR + "/dataset/" + model_name + "/" + mn_version + "/"
    test_dataset_path = ROOT_DIR + "/dataset/fd_test/td_dataset_tabular.csv"

