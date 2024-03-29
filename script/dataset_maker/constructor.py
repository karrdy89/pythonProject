# *********************************************************************************************************************
# Program Name : constructor
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import yaml
import importlib

from VO.request_vo import MakeDataset
from dataset_maker.arg_types import BasicTableType
from dataset_maker.exceptions import *
from statics import ROOT_DIR


def construct_operator(args: MakeDataset) -> BasicTableType:
    """
    Construct inputs for dataset maker with given information
    :param args: MakeDataset
        The pydantic model for creating datasets
    :return: BasicTableType
        The pydantic model for BasicTableType input
    """
    operator_input = {}
    model_id = args.MDL_ID
    main_version = args.MN_VER
    sub_version = args.N_VER
    name = model_id + ":" + main_version + '.' + sub_version

    operator_input["dataset_name"] = model_id
    operator_input["user_id"] = args.USR_ID
    operator_input["actor_name"] = name
    operator_input["start_dtm"] = args.STYMD
    operator_input["end_dtm"] = args.EDYMD
    operator_input["version"] = main_version
    operator_input["num_data_limit"] = int(args.LRNG_DATA_TGT_NCNT)

    try:
        path_dataset_definition = ROOT_DIR + "/script/dataset_maker/dataset_definitions.yaml"
        definitions = get_dataset_definition(path_dataset_definition)
    except Exception as exc:
        raise exc

    definition_list = definitions.get("dataset_definitions", '')
    if definition_list == '':
        raise DefinitionNotFoundError()
    for definition in definition_list:
        if definition.get("name") == model_id:
            try:
                definition = definition.get("definition")
            except Exception as exc:
                raise DefinitionNotFoundError(exc.__str__())
            else:
                try:
                    operator_info = definition.get("operator")
                    operator_input["query_name"] = definition.get("query_name")
                    operator_input["key_index"] = definition.get("key_index")
                    operator_input["feature_index"] = definition.get("feature_index")
                    operator_input["labels"] = definition.get("labels")
                except Exception as exc:
                    raise SetDefinitionError(exc.__str__())
                else:
                    sep_operator_info = operator_info.rsplit('.', 1)
                    module = importlib.import_module(sep_operator_info[0])
                    operator = getattr(module, sep_operator_info[1])
                    try:
                        dataset_maker = operator.options(name=name, max_concurrency=1000).remote()
                    except ValueError as exc:
                        raise SetDefinitionError(exc.__str__())
                    except Exception as exc:
                        raise SetDefinitionError(exc.__str__())
                    else:
                        operator_input["actor_name"] = name
                        operator_input["actor_handle"] = dataset_maker
                        return BasicTableType(**operator_input)
    else:
        raise DefinitionNotExistError()


def get_dataset_definition(path: str) -> dict:
    """
    Parse yaml file with given path
    :param path: str
    :return: dict
    """
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except Exception as exc:
            raise exc
