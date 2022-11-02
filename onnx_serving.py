import os

import ray
from ray import actor
import onnx
import onnxruntime as rt
import numpy as np

from statics import ROOT_DIR
from utils.common import version_encode


@ray.remote
class OnnxServing:
    def __init__(self):
        self._worker: str = type(self).__name__
        self._model_path: str = ""
        self._metadata: dict | None = None
        self._input_type: str = "float"
        self._input_shape: list | None = None
        self._labels: dict | None = None
        self._session = None

    def init(self, model_id: str, version: str) -> int | tuple:
        encoded_version = version_encode(version)
        self._model_path = ROOT_DIR + "/saved_models/" + model_id + "/" + str(encoded_version)
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

        return 0

    def _load_model(self):
        self._metadata = eval(onnx.load(self._model_path).metadata_props[0].value)
        self._labels = self._metadata.get("labels")
        self._input_type = self._metadata.get("input_type")
        self._input_shape = self._metadata.get("input_shape")
        self._session = rt.InferenceSession(self._model_path)

    def predict(self, data: list) -> dict:
        result = {"CODE": "FAIL", "ERROR_MSG": "N/A", "EVNT_ID": [], "PRBT": []}
        if len(data) < self._input_shape[-1]:
            result["CODE"] = "FAIL"
            result["ERROR_MSG"] = "input shape is incorrect"
            return result
        data = data[:self._input_shape[-1]]
        try:
            pred_onx = self._session.run(None, {"input":  np.array([data]).astype(np.float32)})
        except Exception as exc:
            result["CODE"] = "FAIL"
            result["ERROR_MSG"] = "an error occur when get inference from onnx session : " + exc.__str__()
            return result
        pred = pred_onx[0]
        pred_proba = pred_onx[-1][0]
        output_class = []
        output_proba = []
        for vals in pred:
            output_class.append(self._labels.get(vals))
            output_proba.append(pred_proba.get(vals))

        result["CODE"] = "SUCCESS"
        result["ERROR_MSG"] = ""
        result["EVNT_ID"] = output_class
        result["PRBT"] = output_proba
        return result


# onnx_serving = OnnxServing.remote()
# ray.get(onnx_serving.init.remote(model_id="MDL0000002", version="1.1"))
# ray.get(onnx_serving.predict.remote(data=[*range(50)]))