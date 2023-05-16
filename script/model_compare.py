import re
import os
import sys
from datetime import datetime

import pandas as pd

from db import DBUtil
from statics import ROOT_DIR
from utils import version_encode


pattern = re.compile("\d{1,9}[.]\d{1,9}")
version = None

while True:
    version = input("source model version: ")
    if pattern.match(version):
        break
    else:
        print("wrong input. type {mn_ver}.{n_ver} Ex. 1.1")
source_version = version
version = version.split('.')
mn_ver = version[0]
n_ver = version[1]
db = DBUtil(db_info="MANAGE_DB")
data = db.select(name="select_pred_log", param={"MN_VER": mn_ver, "N_VER": n_ver})
if len(data) == 0:
    print("There is no data retrieved for that version.")
else:
    df = pd.DataFrame(data, columns=["SUMN_MSG", "RSLT_MSG"])
    while True:
        version = input("target model version: ")
        if pattern.match(version):
            break
        else:
            print("wrong input. type {mn_ver}.{n_ver} Ex. 1.1")
    target_version = version
    model_key = "MDL0000001_" + version
    b_path = ROOT_DIR + "/saved_models/" + model_key + "/MDL0000001"
    model_path = b_path + "/" + str(version_encode(version))
    if os.path.isdir(model_path):
        if not any(file_name.endswith('.pb') for file_name in os.listdir(model_path)):
            print("There is no model for that version.")
        else:
            import tensorflow as tf

            model = tf.saved_model.load(model_path)
            infer = model.signatures["serving_default"]

            source_result = eval(df.iloc[0]["RSLT_MSG"])
            source_pred = source_result["RSLT"]
            keys = ["version"]
            keys_pred = []
            keys_proba = []
            for i in range(len(source_pred)):
                keys_pred.append("pred_" + str(i))
                keys_proba.append("proba_" + str(i))
            keys = keys + keys_pred + keys_proba

            source_df = pd.DataFrame(columns=keys)
            target_df = pd.DataFrame(columns=keys)
            for index, row in df.iterrows():
                print(str(index) + "/" + str(df.shape[0]), end="\r")
                x = eval(row["SUMN_MSG"])
                source_result = eval(row["RSLT_MSG"])
                source_pred = source_result["RSLT"]
                source_proba = source_result["PRBT"]

                target_result = infer(tf.constant([x]))
                target_pred = target_result["result"].numpy().tolist()
                target_pred = [arg.decode('ascii') for arg in target_pred]
                target_proba = target_result["result_1"].numpy().tolist()

                try:
                    source_df.loc[len(source_df)] = [source_version] + source_pred + source_proba
                    target_df.loc[len(target_df)] = [target_version] + target_pred + target_proba
                except ValueError as exc:
                    print("model output didn't matched:" + exc.__str__())
                    sys.exit()
            else:
                res_df = pd.concat([source_df, target_df], axis=1)
                now = datetime.now()
                date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
                res_df.to_csv(f"./MDL0000001_{source_version}_vs_{target_version}_{date_time}.csv",
                              sep=",", na_rep="NaN", index=False)
    else:
        print("There is no model for that version.")
