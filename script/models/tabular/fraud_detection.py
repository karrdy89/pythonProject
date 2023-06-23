# *********************************************************************************************************************
# Program Name : fraud_detection
# Creator : yum kiyeon
# Create Date : 2022. 11. 10
# Modify Desc :
# *********************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------
# Date  | Updator   | Remark
#
# ---------------------------------------------------------------------------------------------------------------------


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1. refine dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import pandas as pd
from tqdm import tqdm

ROOT_DIR = "/home/ky/PycharmProjects/pythonProject/script"
dataset_path = ROOT_DIR + "/dataset/fd_test2/combined.csv"

df = pd.concat(map(pd.read_csv, [ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_1.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_2.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_3.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_4.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_5.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_6.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_7.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_8.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_9.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_10.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_11.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_12.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_13.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_14.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_15.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_16.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_17.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_18.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_19.csv",
                                 ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_20.csv"]), ignore_index=True)
seq = []
b_seq = 0
fc = 0
for index, row in tqdm(df.iterrows()):
    seq.append(row["SEQ"] + fc*1000)
    if b_seq > row['SEQ']:
        fc += 1
    b_seq = row['SEQ']
print(seq)
df_s = pd.DataFrame(seq, columns=['SEQ'])
df["SEQ"] = df_s
df.to_csv(ROOT_DIR + "/dataset/fd_test2/combined.csv", index=False, encoding="utf-8-sig")
print("done")


# check duplicated seq
df = pd.read_csv(dataset_path)
df["고객ID"].fillna(-1, inplace=True)
df["고객ID"] = df["고객ID"].astype('int32')
sequences = df["SEQ"].unique().tolist()
df_sep_seq = []
for sequence in sequences:
    df_sep_seq.append(df[df["SEQ"] == sequence].reset_index(drop=True))

print(len(df_sep_seq))
dup_cust_ids = []
for idx, df_seq in enumerate(df_sep_seq):
    print(f"{idx} / {len(df_sep_seq)}")
    cust_id_list = df_seq["고객ID"].unique().tolist()
    for t_df_seq in df_sep_seq[idx + 1:]:
        for cust_id in cust_id_list:
            if cust_id == -1:
                continue
            elif cust_id in t_df_seq["고객ID"].values:
                dup_cust_ids.append([df_seq["SEQ"].iloc[0], t_df_seq["SEQ"].iloc[0], cust_id])
                break


print(len(dup_cust_ids))
duped = pd.DataFrame(dup_cust_ids, columns=["SEQ", "SEQ_DUP", "CUST_ID"])
duped.to_csv(ROOT_DIR + "/dataset/fd_test2/combined_duped.csv", index=False, encoding="utf-8-sig")

# drop duplicate
drop_count = 0
for dupe in dup_cust_ids:
    print(f"{drop_count} / {len(dup_cust_ids)}")
    seq_be_del = dupe[1]
    if seq_be_del < len(df_sep_seq):
        df.drop(df[df.SEQ == seq_be_del].index, inplace=True)
        drop_count += 1
print(drop_count)

# drop extra channel
searchfor = ['CCM', 'CCW']
df = df[df.메뉴.str.contains('|'.join(searchfor))]
df.to_csv(ROOT_DIR + "/dataset/fd_test2/combined_trimmed.csv", index=False, encoding="utf-8-sig")
print(len(df["SEQ"].unique().tolist()))
print("done")

# split train and test
# all_dataset_trimmed = combined_trimmed + fraud_dataset_refine_fixed(from first delivered log)
df_all = [pd.read_csv(ROOT_DIR + "/dataset/fd_test2/combined_trimmed.csv"), pd.read_csv(ROOT_DIR + "/dataset/fd_test2/fraud_dataset_refine_fixed.csv.csv")]
df = pd.concat(df_all)
df_n = df[df["정상여부"] == "정상"].reset_index(drop=True)
df_p = df[df["정상여부"] == "금융사기"].reset_index(drop=True)

seq_df_n = df_n["SEQ"].unique().tolist()
len(seq_df_n)
test_list = []

seq_df_p = df_p["SEQ"].unique().tolist()
len(seq_df_p)

import random
train_n = random.sample(range(len(seq_df_n)), len(seq_df_n)-13000)
train_len = len(train_n)

for i, idx in enumerate(train_n):
    print(i, train_len)
    test_list.append(df_n[df_n["SEQ"] == seq_df_n[idx]].reset_index(drop=True))
    df = df[df["SEQ"] != seq_df_n[idx]].reset_index(drop=True)

len(test_list)
len(df["SEQ"].unique().tolist())
test_df = pd.concat(test_list)
df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_train.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_test.csv", index=False, encoding="utf-8-sig")


dataset_path = ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_test.csv"
df = pd.read_csv(dataset_path)
df.fillna("", inplace=True)
df["EVENT"] = df.메뉴.str.cat(df.프로그램명)
df['시간'] = pd.to_datetime(df['시간'], errors='coerce')
df['DATE'] = df['시간'].dt.strftime("%Y-%m-%d")
n_df_list = []

sequences = df["SEQ"].unique().tolist()

df_sep_seq = []
for sequence in sequences:
    df_sep_seq.append(df[df["SEQ"] == sequence].reset_index(drop=True))
start_events = ["CCMLO0101", "CCWLO0101", "CCMMS0101SL01", "CCWMS0101SL01", "CCWSA0101", "CCMSA0101"]
end_event = ["CCMLN0101PC01", "CCWLN0101PC01", "CCWRD0201PC01", "CCMRD0201PC01"]

count = 0
num_multi_cust_id = 0

for idx, df_seq in enumerate(df_sep_seq):
    print(f"{idx} / {len(df_sep_seq)}", end='\r')
    # sep by date
    dates = df_seq["DATE"].unique().tolist()
    df_sep_date_list = []
    for date in dates:
        df_sep_date_list.append(df_seq[df_seq["DATE"] == date].reset_index(drop=True))

    for df_sep_date in df_sep_date_list:
        # sep by ip
        ips = df_sep_date["로그인IP"].unique().tolist()
        df_sep_ip_list = []
        for ip in ips:
            df_sep_ip_list.append(df_sep_date[df_sep_date["로그인IP"] == ip].reset_index(drop=True))
        for df_sep_ip in df_sep_ip_list:
            if len(df_sep_ip["고객ID"].unique().tolist()) > 2:
                num_multi_cust_id += 1
                continue
            start_idx_list = df_sep_ip.index[df_sep_ip["EVENT"].isin(start_events)].tolist()
            end_idx_list = df_sep_ip.index[df_sep_ip["EVENT"].isin(end_event)].tolist()
            if len(start_idx_list) != 0 and len(end_idx_list) != 0:
                start_idx = start_idx_list[0]
                for end_idx in end_idx_list:
                    count += 1
                    temp_df = df_sep_ip[start_idx:end_idx + 1]
                    temp_df.loc[:, "SEQ"] = count
                    n_df_list.append(temp_df.copy())

print(count)

print("done")
print(num_multi_cust_id)
print(len(n_df_list))


n_df = pd.concat(n_df_list, ignore_index=True)
n_df["SEQ"].unique()
n_df["로그인유형"].replace("", "N/A")
assert not n_df.isnull().values.any()
n_df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_test_refine.csv", index=False, encoding="utf-8-sig")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2. transform dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# make dataset to tabular
import pandas as pd


dataset_path = "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine.csv"
df = pd.read_csv(dataset_path)

df.drop(['메뉴', '메뉴명', "고객ID", "로그인IP", "프로그램명", "금액", "로그인유형", "DATE"], axis=1, inplace=True)
classes = {"정상": 0, "금융사기": 1}
df["정상여부"] = df["정상여부"].apply(classes.get).astype(int)

events = df["EVENT"].unique().tolist()
print(len(events))
tabular_dataset = pd.DataFrame(columns=["SEQ"] + events + ["label"])

# separate by seq
seqs = df["SEQ"].unique().tolist()
tdfs = []
for seq in seqs:
    tdfs.append(df[df["SEQ"] == seq].reset_index(drop=True))

import time
import datetime

total = len(tdfs)
for _, tdf in enumerate(tdfs):
    print(_ + 1, "/", total, end='\r')
    seq = tdf["SEQ"].iloc[0]
    label = tdf["정상여부"].iloc[0]
    time_diff = time.mktime(datetime.datetime.strptime(tdf["시간"].iloc[-1], "%Y-%m-%d %H:%M:%S").timetuple()) - \
                time.mktime(datetime.datetime.strptime(tdf["시간"].iloc[0], "%Y-%m-%d %H:%M:%S").timetuple())
    counted_items = (dict(tdf["EVENT"].value_counts()))
    data_row = {"SEQ": seq, "label": label, "time_diff": time_diff}
    data_row.update(counted_items)
    data_row = pd.DataFrame([data_row])
    tabular_dataset = pd.concat([tabular_dataset, data_row], ignore_index=True)

tabular_dataset.to_csv(
    "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular.csv",
    sep=",", index=False, encoding="utf-8-sig")

len(tabular_dataset["SEQ"].unique().tolist())
print("done")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 3. event mapping
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import pandas as pd
mapping_table_path = "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/event_mapping.txt"
mapping_table = pd.read_csv(mapping_table_path, sep="|", header=None)
mapping_dict = dict(zip(mapping_table[0], mapping_table[2]))
mapping_dict["SEQ"] = "SEQ"
mapping_dict["label"] = "label"
mapping_dict["time_diff"] = "time_diff"

train_df_path = "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular.csv"
train_df = pd.read_csv(train_df_path)
old_cols = train_df.keys().tolist()

col_to_drop = []
mapped_col = []

for i, col in enumerate(old_cols):
    new_col = mapping_dict.get(col)
    if new_col is None:
        col_to_drop.append(col)
    else:
        mapped_col.append(new_col)

train_df = train_df.drop(col_to_drop, axis=1)
train_df.columns = mapped_col
train_df = train_df.groupby(level=0, axis=1).sum()

train_df.to_csv("/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_mapped.csv", sep=",", index=False, encoding="utf-8-sig")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 4. feature engineering
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# feature construction
import pandas as pd

dataset_path = "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_mapped.csv"
df = pd.read_csv(dataset_path)
df.fillna(0, inplace=True)
df_abnormal = df[df["label"] == 1].reset_index(drop=True)

nc_cols = []
for idx, column in enumerate(df_abnormal):
    if column != "label" and column != "SEQ":
        if df_abnormal[column].sum() == 0:
            nc_cols.append(column)

nc_df = df.loc[:, nc_cols]
df["ETC"] = nc_df.sum(axis="columns")
df.drop(nc_cols, axis=1, inplace=True)
cols = df.columns.tolist()
cols = cols[:-2] + [cols[-1]] + [cols[-2]]
df = df[cols]
df.to_csv("/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_mapped_fc.csv", sep=",", index=False, encoding="utf-8-sig")
print("done")


# feature selection
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

dataset_path = "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_mapped_fc.csv"
df = pd.read_csv(dataset_path)
df_labels = df[["label"]]
dft = df.drop(["SEQ", "label", "ETC"], axis=1)
feature_list = dft.keys().tolist()
X = np.array(dft.values.tolist())
y = np.array(df_labels.values.tolist()).ravel()

model = XGBClassifier(learning_rate=0.03,
                      colsample_bytree=1,
                      subsample=1,
                      objective='binary:logistic',
                      n_estimators=2000,
                      reg_alpha=0.25,
                      max_depth=4,
                      scale_pos_weight=10,
                      gamma=0.3)
pipe = Pipeline([('scaler', RobustScaler()),
                 ('rf_classifier', model)])

pipe.fit(X, y)
pred_x = pipe.predict(X)

print(f1_score(pred_x, y, average='binary', zero_division=1))
print(roc_auc_score(pred_x, y))
conf_matrix = confusion_matrix(y_true=y, y_pred=pred_x)
print(conf_matrix.ravel())
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

XGB_importances = pd.Series(importance, index=feature_list)
npc = XGB_importances.loc[XGB_importances[:] == 0]
npc = npc.keys().tolist()
print("feature ranking")
for i in range(len(feature_list)):
    print("{}. feature {} ({:.3f}) ".format(i+1, feature_list[indices[i]], importance[indices[i]]))
print(len(npc))
df.drop(npc, axis=1, inplace=True)
df.to_csv("/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_mapped_fc_fe.csv",
          sep=",", index=False, encoding="utf-8-sig")


# make test dataset
import pandas as pd
dataset_path = "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_test_refine.csv"
keys = pd.read_csv("/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_mapped_fc_fe.csv").keys().tolist()
keys.remove("SEQ")
keys.remove("label")
keys.remove("ETC")
keys.remove("time_diff")

df.drop(['메뉴', '메뉴명', "고객ID", "로그인IP", "프로그램명", "금액", "로그인유형", "DATE"], axis=1, inplace=True)
classes = {"정상": 0, "금융사기": 1}
df["정상여부"] = df["정상여부"].apply(classes.get).astype(int)

mapping_table_path = "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/event_mapping.txt"
mapping_table = pd.read_csv(mapping_table_path, sep="|", header=None)
mapping_dict = dict(zip(mapping_table[0], mapping_table[2]))
df["EVENT"].replace(mapping_dict, inplace=True)

selected_event = {}
for key in keys:
    selected_event[key] = 0

events = list(selected_event.keys())
print(len(events))
tabular_dataset = pd.DataFrame(columns=["SEQ"] + events + ["label"])

# separate by seq
seqs = df["SEQ"].unique().tolist()
tdfs = []
for seq in seqs:
    tdfs.append(df[df["SEQ"] == seq].reset_index(drop=True))

import time
import datetime

total = len(tdfs)
for _, tdf in enumerate(tdfs):
    print(_ + 1, "/", total, end='\r')
    seq = tdf["SEQ"].iloc[0]
    label = tdf["정상여부"].iloc[0]
    time_diff = time.mktime(datetime.datetime.strptime(tdf["시간"].iloc[-1], "%Y-%m-%d %H:%M:%S").timetuple()) - \
                time.mktime(datetime.datetime.strptime(tdf["시간"].iloc[0], "%Y-%m-%d %H:%M:%S").timetuple())
    counted_items = (dict(tdf["EVENT"].value_counts()))
    # select selected_event only
    tmp_items = {}
    for k, v in selected_event.items():
        if k in counted_items:
            tmp_items[k] = counted_items[k]
        else:
            tmp_items[k] = v
    counted_items = tmp_items

    data_row = {"SEQ": seq, "label": label, "time_diff": time_diff}
    data_row.update(counted_items)
    data_row = pd.DataFrame([data_row])
    tabular_dataset = pd.concat([tabular_dataset, data_row], ignore_index=True)

tabular_dataset.to_csv(
    "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_test_refine_mapped.csv",
    sep=",", index=False, encoding="utf-8-sig")

len(tabular_dataset["SEQ"].unique().tolist())
print("done")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 5. oversampling & train
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import os
import random
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import ADASYN, SMOTE
from xgboost import XGBClassifier
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType

train_dataset_path = "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_mapped_fc_fe.csv"
train_df = pd.read_csv(train_dataset_path)
train_df_labels = train_df[["label"]]
train_df_t = train_df.drop(["SEQ", "label", "ETC"], axis=1)


test_dataset_path = "/home/ky/PycharmProjects/pythonProject/dataset/fd_test2/all_dataset_trimmed_test_refine_mapped.csv"
test_df = pd.read_csv(test_dataset_path)
test_df_labels = test_df[["label"]]
test_df_t = test_df.drop(["SEQ", "label"], axis=1)
feature_list = test_df_t.keys().tolist()
X_test = np.array(test_df_t.values.tolist())
y_test = np.array(test_df_labels.values.tolist()).ravel()

train_df_t = train_df_t[feature_list]
print(feature_list)
print(train_df_t)
X_train = np.array(train_df_t.values.tolist())
y_train = np.array(train_df_labels.values.tolist()).ravel()
original_data_idx = len(X_train)
num_neg = len(train_df[train_df["label"] == 1])
num_pos = len(X_train[:-num_neg])


hp_resample_count = 10
hp_grid_rl = [0.2, 0.3, 0.25, 0.15]
hp_grid_estimators = [800, 500, 1000, 1200, 1500, 1800]
hp_grid_m_depth = [5, 4]
hp_grid_spw = [100, 50, 10, 200, 250]
hp_grid_gamma = [0.25, 0.35, 0.3]

log = {}
for i in range(hp_resample_count):
    print("resample")
    num_neg = len(train_df[train_df["label"] == 1])
    sm = SMOTE(random_state=1, sampling_strategy=0.10, k_neighbors=5)
    ad = ADASYN(random_state=1 + 1, sampling_strategy=0.10)
    sm2 = SMOTE(random_state=1 + 2, sampling_strategy=0.03)

    X_ovs_stack_ad, y_ovs_stack_ad = ad.fit_resample(X_train, y_train)
    for i, org in enumerate(X_train[-num_neg:]):
        assert np.array_equal(org,
                              X_ovs_stack_ad[original_data_idx - num_neg:original_data_idx][i]), "slice ad test"

    X_ovs_stack_sm, y_ovs_stack_sm = sm.fit_resample(X_train, y_train)
    for i, org in enumerate(X_train[-num_neg:]):
        assert np.array_equal(org,
                              X_ovs_stack_sm[original_data_idx - num_neg:original_data_idx][i]), "slice sm test"
    X_ovs_stack_sm = X_ovs_stack_sm[original_data_idx:]
    y_ovs_stack_sm = y_ovs_stack_sm[original_data_idx:]

    X_ovs_stack_smt, y_ovs_stack_smt = sm2.fit_resample(X_train, y_train)
    for i, org in enumerate(X_train[-num_neg:]):
        assert np.array_equal(org,
                              X_ovs_stack_smt[original_data_idx - num_neg:original_data_idx][i]), "slice sm2 test"
    X_ovs_stack_smt = X_ovs_stack_smt[original_data_idx:]
    y_ovs_stack_smt = y_ovs_stack_smt[original_data_idx:]

    X_ovs_stacked = np.concatenate([X_ovs_stack_ad, X_ovs_stack_sm, X_ovs_stack_smt])
    y_ovs_stacked = np.concatenate([y_ovs_stack_ad, y_ovs_stack_sm, y_ovs_stack_smt])

    X_ovs_pos = X_ovs_stacked[:original_data_idx - num_neg]
    X_ovs_neg = X_ovs_stacked[original_data_idx - num_neg:]
    y_ovs_pos = y_ovs_stacked[:original_data_idx - num_neg]
    y_ovs_neg = y_ovs_stacked[original_data_idx - num_neg:]

    print(int(len(X_ovs_neg) * 0.1))
    test_n = random.sample(range(len(X_ovs_neg)), int(len(X_ovs_neg) * 0.1))
    test_n.sort(reverse=True)
    test_len = len(test_n)
    test_neg_X = []
    test_neg_y = []
    for i, idx in enumerate(test_n):
        print(i, test_len, end='\r')
        test_neg_X.append(X_ovs_neg[idx].tolist())
        test_neg_y.append(y_ovs_neg[idx].tolist())
        X_ovs_neg = np.delete(X_ovs_neg, idx, axis=0)
        y_ovs_neg = np.delete(y_ovs_neg, idx, axis=0)

    X_ovs_train = np.concatenate([X_ovs_pos, X_ovs_neg])
    y_ovs_train = np.concatenate([y_ovs_pos, y_ovs_neg])
    X_ovs_test = np.concatenate([X_test, np.array(test_neg_X)])
    y_ovs_test = np.concatenate([y_test, test_neg_y])

    class_total = Counter(y_ovs_train)
    test_class_total = Counter(y_ovs_test)
    print('Resampled dataset shape %s' % class_total)
    print('Resampled test dataset shape %s' % test_class_total)

    for rl in hp_grid_rl:
        print("rl")
        for estimator in hp_grid_estimators:
            print("est")
            for m_depth in hp_grid_m_depth:
                print("grd_depth")
                for gamma in hp_grid_gamma:
                    print("grd_gamma")
                    for spw in hp_grid_spw:
                        print("grd_spw")
                        model = XGBClassifier(learning_rate=rl,
                                              colsample_bytree=1,
                                              subsample=0.75,
                                              objective='binary:logistic',
                                              n_estimators=estimator,
                                              reg_alpha=0.5,
                                              max_depth=m_depth,
                                              scale_pos_weight=spw,
                                              gamma=gamma)

                        pipe = Pipeline([('scaler', RobustScaler()),
                                         ('rf_classifier', model)])

                        pipe.fit(X_ovs_train, y_ovs_train)

                        # get test result
                        pred_all = pipe.predict(X_ovs_train)
                        pred_all_proba = pipe.predict_proba(X_ovs_train)
                        conf_matrix = confusion_matrix(y_true=y_ovs_train, y_pred=pred_all)
                        conf_matrix = list([int(x) for x in conf_matrix.ravel()])
                        f1 = f1_score(y_ovs_train, pred_all, average='binary', zero_division=1)
                        roc_auc = roc_auc_score(y_ovs_train, pred_all)
                        print(f"train confusion matrix: FP:{conf_matrix[1]}, FN:{conf_matrix[2]}")
                        if conf_matrix[1] < 1 and conf_matrix[2] < 1:
                            print("train score: ", f"f1 score:{f1}", f"roc score: {roc_auc}")
                            pred_test = pipe.predict(X_ovs_test)
                            pred_proba_test = pipe.predict_proba(X_ovs_test)
                            # check f1_score in test set
                            conf_matrix_test = confusion_matrix(y_true=y_ovs_test, y_pred=pred_test)
                            conf_matrix_test = list([int(x) for x in conf_matrix_test.ravel()])
                            print(f"test confusion matrix: FP:{conf_matrix_test[1]}, FN:{conf_matrix_test[2]}")
                            if conf_matrix_test[1] < 1 and conf_matrix_test[2] < 1:
                                f1_test = f1_score(y_ovs_test, pred_test, average='binary', zero_division=1)
                                roc_auc_test = roc_auc_score(y_ovs_test, pred_test)
                                print("test score: ", f"f1 score:{f1_test}", f"roc score: {roc_auc_test}")

                                # save best model as-is
                                import joblib
                                filename = "/home/ky/PycharmProjects/pythonProject/b/fd_xgboost_ov_pipe.sav"
                                joblib.dump(pipe, filename)

                                update_registered_converter(
                                    XGBClassifier, 'XGBoostXGBClassifier',
                                    calculate_linear_classifier_output_shapes, convert_xgboost,
                                    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

                                model_onnx = convert_sklearn(pipe, 'pipeline_xgb',
                                                             [(
                                                              'input', FloatTensorType([None, len(feature_list)]))])

                                # add metadata and save model as onnx
                                meta = model_onnx.metadata_props.add()
                                meta.key = "model_info"
                                cfg = {"input_type": "float", "input_shape": [None, len(feature_list)],
                                       "labels": {0: "정상", 1: "전자금융피해"},
                                       "transformer": "fraud_detection.transform_data",
                                       "threshold": 0.5,
                                       "pos_class": 1}
                                meta.value = str(cfg)

                                from datetime import datetime
                                import json

                                now = datetime.now()
                                date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
                                saved_model_path = "/home/ky/PycharmProjects/pythonProject/saved_models/td_test/" + date_time + "/"
                                log["info"] = {"threshold": 0.5,
                                               "hp_grid_gamma": gamma,
                                               "hp_grid_spw": spw,
                                               "hp_grid_m_depth": m_depth,
                                               "hp_grid_estimators": estimator,
                                               "hp_grid_rl": rl,
                                               "TEST_FP": conf_matrix_test[1],
                                               "TEST_FN": conf_matrix_test[2],
                                               "F1": f1,
                                               "ROC_AUC": roc_auc,
                                               "F1_TEST": f1_test,
                                               "ROC_AUC_TEST": roc_auc_test}
                                if not os.path.exists(saved_model_path):
                                    os.makedirs(saved_model_path)
                                with open(saved_model_path + "fd_xgboost_ov.onnx", "wb") as f:
                                    f.write(model_onnx.SerializeToString())
                                with open(saved_model_path + "/Hparams.json", 'w', encoding="utf-8") as f:
                                    json.dump(log, f, ensure_ascii=False, indent=4)
