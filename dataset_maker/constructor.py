import yaml
import importlib

from VO.request_vo import MakeDataset
from dataset_maker.arg_types import BasicTableType
from dataset_maker.exceptions import *
from statics import BuiltinModels, ROOT_DIR


def construct_operator(args: MakeDataset) -> BasicTableType:
    operator_input = {}
    model_id = args.MDL_ID
    model_name = getattr(BuiltinModels, model_id)
    model_name = model_name.model_name
    main_version = args.MN_VER
    sub_version = args.N_VER
    name = model_id + ":" + main_version + '.' + sub_version

    operator_input["dataset_name"] = model_id
    operator_input["actor_name"] = name
    operator_input["start_dtm"] = args.STYMD
    operator_input["end_dtm"] = args.EDYMD
    operator_input["version"] = main_version
    operator_input["num_data_limit"] = int(args.LRNG_DATA_TGT_NCNT)

    try:
        path_dataset_definition = ROOT_DIR + "/" + "dataset_maker" + "/dataset_definitions.yaml"
        definitions = get_dataset_definition(path_dataset_definition)
    except Exception as exc:
        raise exc

    definition_list = definitions.get("dataset_definitions", '')
    if definition_list == '':
        raise DefinitionNotFoundError()
    for definition in definition_list:
        if definition.get("name") == model_name:
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
                        dataset_maker = operator.options(name=name).remote()
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
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except Exception as exc:
            raise exc
