# *********************************************************************************************************************
# Program Name : onnx_serving
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------
import importlib
import os
import random

import ray
import onnx
import onnxruntime as rt
import numpy as np

from statics import ROOT_DIR
from utils.common import version_encode


@ray.remote
class OnnxServing:
    """
    A ray actor class for serve onnx model

    Attributes
    ----------
    _worker : str
        The class name of instance.
    _model_path : str
        The path of model directory.
    _metadata: dict | None
        The metadata of model
    _input_type: str | None
        Input type of model defined in metadata
    _input_shape: str | None
        Input shape of model defined in metadata
    _labels: str | None
        Class name of model output defined in metadata
    _session:
        The onnx inference session

    Methods
    -------
    __init__():
        Constructs all the necessary attributes.
    _load_model() -> int
        Load model and set metadata
    predict(data: list) -> dict:
        Return predict result
    """

    def __init__(self):
        self._worker: str = type(self).__name__
        self._model_path: str = ""
        self._metadata: dict | None = None
        self._input_type: str | None = None
        self._input_shape: list | None = None
        self._transformer: str | None = None
        self._threshold: int | None = None
        self._pos_class: int | None = None
        self._labels: dict | None = None
        self._session = None

    def init(self, model_id: str, version: str) -> tuple:
        encoded_version = version_encode(version)
        model_key = model_id + "_" + version
        self._model_path = ROOT_DIR + "/saved_models/" + model_key + "/" + model_id + "/" + str(encoded_version)
        model_name = None
        for folderName, subfolders, filenames in os.walk(self._model_path):
            for filename in filenames:
                ext = filename.split('.')[-1]
                if ext == "onnx":
                    model_name = filename
                    self._model_path = self._model_path + "/" + model_name
                    break
        else:
            if model_name is None:
                return -1, "model not found"

        try:
            self._load_model()
        except Exception as exc:
            return -1, exc.__str__()
        else:
            return 0, "success"

    def _load_model(self):
        try:
            self._metadata = eval(onnx.load(self._model_path).metadata_props[0].value)
            self._labels = self._metadata.get("labels")
            self._input_type = self._metadata.get("input_type")
            self._input_shape = self._metadata.get("input_shape")
            self._transformer = self._metadata.get("transformer")
            self._threshold = self._metadata.get("threshold")
            self._pos_class = self._metadata.get("pos_class")
        except Exception as exc:
            print(self._worker + ": " + exc.__str__())
        self._session = rt.InferenceSession(self._model_path)

    def predict(self, data: list) -> tuple | dict:
        result = {"CODE": "FAIL", "ERROR_MSG": "N/A", "RSLT": []}
        if self._transformer is not None:
            sp_transformer_info = self._transformer.split('.')
            module = ''.join(sp_transformer_info[:-1])
            if module == "fraud_detection":
                # module = importlib.import_module(module)
                # func = sp_transformer_info[-1]
                # func = getattr(module, func)
                # data = func(data)
                data = [random.randrange(0, 5)] * 46
        if self._input_shape is not None:
            if len(data) < self._input_shape[-1]:
                result["CODE"] = "FAIL"
                result["ERROR_MSG"] = "input shape is incorrect"
                return result, data
            elif len(data) == 0:
                result["CODE"] = "FAIL"
                result["ERROR_MSG"] = "input vector is empty"
                return result, data
            else:
                data = data[:self._input_shape[-1]]
        try:
            if self._input_type != "float":
                pred_onx = self._session.run(None, {"input": np.array([data]).astype(np.object)})
            else:
                pred_onx = self._session.run(None, {"input": np.array([data]).astype(np.float32)})
        except Exception as exc:
            result["CODE"] = "FAIL"
            result["ERROR_MSG"] = "an error occur when get inference from onnx session : " + exc.__str__()
            return result
        pred = pred_onx[0]
        pred_proba = pred_onx[-1][0]
        output_class = []
        output_proba = []
        neg_class = None
        if self._threshold is not None and self._pos_class is not None:
            if pred[0] == self._pos_class:
                if pred_proba.get(self._pos_class) < self._threshold:
                    for key in pred_proba:
                        if key != self._pos_class:
                            neg_class = key
        for vals in pred:
            if self._labels is not None:
                if neg_class is not None:
                    output_class.append(self._labels.get(neg_class))
                else:
                    output_class.append(self._labels.get(vals))
            else:
                if neg_class is not None:
                    output_class.append(neg_class)
                else:
                    output_class.append(vals)
            if neg_class is not None:
                output_proba.append(pred_proba.get(neg_class))
            else:
                output_proba.append(pred_proba.get(vals))
        result["CODE"] = "SUCCESS"
        result["ERROR_MSG"] = ""
        f_res = []
        for i in range(output_class):
            f_res.append({"NAME": output_class[i], "PRBT": output_proba[i]})
        # result["RSLT"] = output_class
        # result["PRBT"] = output_proba
        result["RSLT"] = f_res
        return result, data
