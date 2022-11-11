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

# # trim off garbage data
# import pandas as pd
# from statics import ROOT_DIR

# dataset_path = ROOT_DIR + "/dataset/fd_test/fraud_dataset.csv"
# saved_model_path = ROOT_DIR + "/saved_models/td_test/"
# df = pd.read_csv(dataset_path)
# df["고객ID"].fillna(-1, inplace=True)
# df["고객ID"] = df["고객ID"].astype('int32')
# # check duplicated seq
# sequences = df["SEQ"].unique().tolist()
#
# df_sep_seq = []
# for sequence in sequences:
#     df_sep_seq.append(df[df["SEQ"] == sequence].reset_index(drop=True))
#
# print(len(df_sep_seq))
# dup_cust_ids = []
# for idx, df_seq in enumerate(df_sep_seq):
#     cust_id_list = df_seq["고객ID"].unique().tolist()
#     for t_df_seq in df_sep_seq[idx + 1:]:
#         for cust_id in cust_id_list:
#             if cust_id == -1:
#                 continue
#             elif cust_id in t_df_seq["고객ID"].values:
#                 dup_cust_ids.append([df_seq["SEQ"].iloc[0], t_df_seq["SEQ"].iloc[0], cust_id])
#                 break
#
#
# print(len(dup_cust_ids))
# duped = pd.DataFrame(dup_cust_ids, columns=["SEQ", "SEQ_DUP", "CUST_ID"])
# duped.to_csv(ROOT_DIR + "/dataset/fd_test/duplicated.csv", index=False)

# # drop duplicate
# for dupe in dup_cust_ids:
#     seq_be_del = dupe[1]
#     print(seq_be_del)
#     if seq_be_del < 1000:
#         df.drop(df[df.SEQ == seq_be_del].index, inplace=True)

# # drop extra channel
# searchfor = ['CCM', 'CCW']
# df = df[df.메뉴.str.contains('|'.join(searchfor))]
# df.to_csv(ROOT_DIR + "/dataset/fd_test/fd_dataset_trimmed.csv", index=False)


# # refine data with trimmed dataset
# import pandas as pd
# from statics import ROOT_DIR
# dataset_path = ROOT_DIR + "/dataset/fd_test/fd_dataset_trimmed.csv"
# df = pd.read_csv(dataset_path)
# df.fillna("", inplace=True)
# df["EVENT"] = df.메뉴.str.cat(df.프로그램명)
# df['시간'] = pd.to_datetime(df['시간'], errors='coerce')
# df['DATE'] = df['시간'].dt.strftime("%Y-%m-%d")
#
# sequences = df["SEQ"].unique().tolist()
# df_sep_seq = []
# for sequence in sequences:
#     df_sep_seq.append(df[df["SEQ"] == sequence].reset_index(drop=True))
# start_events = ["CCMLO0101", "CCWLO0101", "CCMMS0101SL01", "CCWMS0101SL01", "CCWSA0101", "CCMSA0101"]
# end_event = ["CCMLN0101PC01", "CCWLN0101PC01", "CCWRD0201PC01", "CCMRD0201PC01"]
#
# count = 0
# num_multi_cust_id = 0
# n_df_list = []
# for idx, df_seq in enumerate(df_sep_seq):
#     # sep by date
#     dates = df_seq["DATE"].unique().tolist()
#     df_sep_date_list = []
#     for date in dates:
#         df_sep_date_list.append(df_seq[df_seq["DATE"] == date].reset_index(drop=True))
#
#     for df_sep_date in df_sep_date_list:
#         # sep by ip
#         ips = df_sep_date["로그인IP"].unique().tolist()
#         df_sep_ip_list = []
#         for ip in ips:
#             df_sep_ip_list.append(df_sep_date[df_sep_date["로그인IP"] == ip].reset_index(drop=True))
#         for df_sep_ip in df_sep_ip_list:
#             if len(df_sep_ip["고객ID"].unique().tolist()) > 2:
#                 num_multi_cust_id += 1
#                 continue
#             start_idx_list = df_sep_ip.index[df_sep_ip["EVENT"].isin(start_events)].tolist()
#             end_idx_list = df_sep_ip.index[df_sep_ip["EVENT"].isin(end_event)].tolist()
#             if len(start_idx_list) != 0 and len(end_idx_list) != 0:
#                 start_idx = start_idx_list[0]
#                 for end_idx in end_idx_list:
#                     count += 1
#                     temp_df = df_sep_ip[start_idx:end_idx + 1]
#                     temp_df.loc[:, "SEQ"] = count
#                     n_df_list.append(temp_df.copy())
#
# print("done")
# print(num_multi_cust_id)
# print(len(n_df_list))
# n_df = pd.concat(n_df_list, ignore_index=True)
# n_df["SEQ"].unique()
# n_df["로그인유형"].replace("", "N/A")
# n_df["비고"].replace("", "N/A")
# assert not n_df.isnull().values.any()
# n_df.to_csv(ROOT_DIR + "/dataset/fd_test/fraud_dataset_refine_fixed.csv", index=False, encoding="utf-8-sig")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2. transform dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# #make dataset to tabular
# import pandas as pd
# from statics import ROOT_DIR
# dataset_path = ROOT_DIR + "/dataset/fd_test/fraud_dataset_refine_fixed.csv"
# df = pd.read_csv(dataset_path)
# df.drop(['메뉴', '메뉴명', "고객ID", "로그인IP", "프로그램명", "시간", "금액", "로그인유형", "비고", "DATE"], axis=1, inplace=True)
# df["EVENT"] = df["EVENT"].str.replace("CCW", "CCM")
# classes = {"정상": 0, "금융사기": 1}
# df["정상여부"] = df["정상여부"].apply(classes.get).astype(int)
#
# events = df["EVENT"].unique().tolist()
# tabular_dataset = pd.DataFrame(columns=["SEQ"] + events + ["label"])
#
# # separate by seq
# seqs = df["SEQ"].unique().tolist()
# tdfs = []
# for seq in seqs:
#     tdfs.append(df[df["SEQ"] == seq].reset_index(drop=True))
#
# total = len(tdfs)
# for _, tdf in enumerate(tdfs):
#     print(_+1, "/", total)
#     seq = tdf["SEQ"].iloc[0]
#     label = tdf["정상여부"].iloc[0]
#     counted_items = (dict(tdf["EVENT"].value_counts()))
#     data_row = {"SEQ": seq, "label": label}
#     data_row.update(counted_items)
#     data_row = pd.DataFrame([data_row])
#     tabular_dataset = pd.concat([tabular_dataset, data_row], ignore_index=True)
#
# tabular_dataset.to_csv(ROOT_DIR + "/dataset/fd_test/fraud_dataset_tabular_fixed.csv", sep=",",
#                        index=False, encoding="utf-8-sig")
#
# print("done")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 3. feature engineering
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # feature construction
# import pandas as pd
# from statics import ROOT_DIR
#
# dataset_path = ROOT_DIR + "/dataset/fd_test/fraud_dataset_tabular_fixed.csv"
# df = pd.read_csv(dataset_path)
# df.fillna(0, inplace=True)
# df_abnormal = df[df["label"] == 1].reset_index(drop=True)
#
# nc_cols = []
# for column in df_abnormal:
#     if column != "label" and column != "SEQ":
#         if df_abnormal[column].sum() == 0:
#             nc_cols.append(column)
#
# nc_df = df.loc[:, nc_cols]
# df["ETC"] = nc_df.sum(axis="columns")
# df.drop(nc_cols, axis=1, inplace=True)
# cols = df.columns.tolist()
# cols = cols[:-2] + [cols[-1]] + [cols[-2]]
# df = df[cols]
# df.to_csv(ROOT_DIR + "/dataset/fd_test/fraud_dataset_tabular_fixed_fc.csv", sep=",", index=False, encoding="utf-8-sig")
# print("done")


# # feature selection
# import pandas as pd
# import numpy as np
# from xgboost import XGBClassifier
# from sklearn.metrics import f1_score
#
# from statics import ROOT_DIR
#
# dataset_path = ROOT_DIR + "/dataset/fd_test/fraud_dataset_tabular_fixed_fc.csv"
# df = pd.read_csv(dataset_path)
# num_testdata = len(df[df["label"] == 1])
# num_neg = num_testdata
# df_labels = df[["label"]]
# dft = df.drop(["SEQ", "label", "ETC"], axis=1)
# feature_list = dft.keys().tolist()
# X = np.array(dft.values.tolist())
# y = np.array(df_labels.values.tolist()).ravel()
#
# # # visualization original
# # from sklearn.manifold import TSNE
# #
# #
# # t_sne_1 = TSNE(n_components=2, perplexity=5, n_iter=3000, learning_rate="auto", init="pca")
# # res_0 = t_sne_1.fit_transform(X[:-num_neg])
# #
# # t_sne_2 = TSNE(n_components=2, perplexity=1, n_iter=3000, learning_rate="auto", init="pca")
# # res_1 = t_sne_2.fit_transform(X[-num_neg:])
# #
# # plt.scatter(res_0[:, 0], res_0[:, 1])
# # plt.scatter(res_1[:, 0], res_1[:, 1])
# # plt.show()
# #
#
# model = XGBClassifier(learning_rate=0.018,
#                       colsample_bytree=1,
#                       subsample=1,
#                       objective='binary:logistic',
#                       n_estimators=2000,
#                       reg_alpha=0.25,
#                       max_depth=4,
#                       scale_pos_weight=250,
#                       gamma=0.22)
#
#
# model.fit(X, y)
# pred_x = model.predict(X)
# print(f1_score(pred_x, y, average='binary', zero_division=1))
# importance = model.feature_importances_
# indices = np.argsort(importance)[::-1]
#
# XGB_importances = pd.Series(importance, index=feature_list)
# npc = XGB_importances.loc[XGB_importances[:] == 0]
# npc = npc.keys().tolist()
# print("feature ranking")
# for i in range(len(feature_list)):
#     print("{}. feature {} ({:.3f}) ".format(i+1, feature_list[indices[i]], importance[indices[i]]))
# print(len(npc))
# df.drop(npc, axis=1, inplace=True)
# df.to_csv(ROOT_DIR + "/dataset/fd_test/fraud_dataset_tabular_fixed_fc_fs.csv",
#           sep=",", index=False, encoding="utf-8-sig")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 4. oversampling & train
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# import os
# from collections import Counter
#
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from numpy import mean
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import RobustScaler
# from sklearn.pipeline import Pipeline
# from sklearn.utils import shuffle
# from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
# from imblearn.over_sampling import ADASYN, SMOTE
# from imblearn.combine import SMOTETomek
# from xgboost import XGBClassifier
# from skl2onnx import convert_sklearn, update_registered_converter
# from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
# from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
# from skl2onnx.common.data_types import FloatTensorType
#
# from statics import ROOT_DIR
#
# dataset_path = ROOT_DIR + "/dataset/fd_test/fraud_dataset_tabular_fixed_fc_fs.csv"
# df = pd.read_csv(dataset_path)
# df_labels = df[["label"]]
# dft = df.drop(["SEQ", "label", "ETC"], axis=1)
# feature_list = dft.keys().tolist()
# X = np.array(dft.values.tolist())
# y = np.array(df_labels.values.tolist()).ravel()
# original_data_idx = len(X)
# num_neg = len(df[df["label"] == 1])
# num_pos = len(X[:-num_neg])
#
#
# def split_train_test(X_array: np.ndarray, y_array: np.ndarray, neg_start_idx: int, sample_num_pos_test: int,
#                      sample_num_neg_test: int, return_test_set: bool):
#     X_origin_data = X_array
#     y_origin_data = y_array
#     X_pos_data = X_origin_data[:-neg_start_idx]
#     y_pos_data = y_origin_data[:-neg_start_idx]
#     y_pos_data = y_pos_data.reshape(y_pos_data.shape[0], -1)
#     X_neg_data = X_origin_data[-neg_start_idx:]
#     y_neg_data = y_origin_data[-neg_start_idx:]
#     y_neg_data = y_neg_data.reshape(y_neg_data.shape[0], -1)
#     pos_indices = np.random.choice(len(X_pos_data)-1, sample_num_pos_test, replace=False)
#     X_pos_test = X_pos_data[pos_indices, :]
#     y_pos_test = y_pos_data[pos_indices, :]
#     X_pos_data = np.delete(X_pos_data, pos_indices, 0)
#     y_pos_data = np.delete(y_pos_data, pos_indices, 0)
#
#     if sample_num_neg_test != -1:
#         neg_indices = np.random.choice(len(X_neg_data)-1, sample_num_pos_test, replace=False)
#         X_neg_test = X_neg_data[neg_indices, :]
#         y_neg_test = y_neg_data[neg_indices, :]
#         X_neg_data = np.delete(X_neg_data, neg_indices, 0)
#         y_neg_data = np.delete(y_neg_data, neg_indices, 0)
#         X_train = np.concatenate([X_pos_data, X_neg_data])
#         y_train = np.concatenate([y_pos_data.ravel(), y_neg_data.ravel()])
#         X_test = np.concatenate([X_pos_test, X_neg_test])
#         y_test = np.concatenate([y_pos_test.ravel(), y_neg_test.ravel()])
#     else:
#         X_neg_test = X_neg_data
#         y_neg_test = y_neg_data
#         X_train = X_pos_data
#         y_train = y_pos_data.ravel()
#         X_test = np.concatenate([X_pos_test, X_neg_test])
#         y_test = np.concatenate([y_pos_test.ravel(), y_neg_test.ravel()])
#     if return_test_set:
#         return X_train, y_train, X_test, y_test
#     else:
#         return X_train, y_train
#
#
# sm = SMOTE(random_state=42, sampling_strategy=0.12, k_neighbors=5)
# ad = ADASYN(random_state=43, sampling_strategy=0.12)
# smt = SMOTETomek(random_state=44, sampling_strategy=0.05)
#
# X_oversampled, y_oversampled = ad.fit_resample(X, y)
#
# for i, org in enumerate(X[-num_neg:]):
#     assert np.array_equal(org, X_oversampled[original_data_idx-num_neg:original_data_idx][i]), "on splice ad test"
#
# X_ovs_stack_sm, y_ovs_stack_sm = sm.fit_resample(X, y)
# for i, org in enumerate(X[-num_neg:]):
#     assert np.array_equal(org, X_ovs_stack_sm[original_data_idx-num_neg:original_data_idx][i]), "on splice sm test"
# X_ovs_stack_sm = X_ovs_stack_sm[original_data_idx:]
# y_ovs_stack_sm = y_ovs_stack_sm[original_data_idx:]
#
# X_ovs_stack_smt, y_ovs_stack_smt = smt.fit_resample(X, y)
# for i, org in enumerate(X[-num_neg:]):
#     assert np.array_equal(org, X_ovs_stack_smt[original_data_idx-num_neg:original_data_idx][i]), "on splice smt test"
# X_ovs_stack_smt = X_ovs_stack_smt[original_data_idx:]
# y_ovs_stack_smt = y_ovs_stack_smt[original_data_idx:]
#
#
# X_ov_train, y_ov_train, X_ov_t_test, y_ov_t_test = \
#     split_train_test(X_oversampled[:original_data_idx], y_oversampled[:original_data_idx], num_neg, int(num_pos/10), -1, True)
# X_ov_train = np.concatenate([X_ov_train, X_oversampled[original_data_idx:]])
# y_ov_train = np.concatenate([y_ov_train, y_oversampled[original_data_idx:]])
#
# X_ov_train_stacked = np.concatenate([X_ov_train, X_ovs_stack_sm, X_ovs_stack_smt])
# y_ov_train_stacked = np.concatenate([y_ov_train, y_ovs_stack_sm, y_ovs_stack_smt])
# X_ov_train_stacked, y_ov_train_stacked = shuffle(X_ov_train_stacked, y_ov_train_stacked)
#
# class_total = Counter(y_ov_train_stacked)
# print('Resampled dataset shape %s' % class_total)
#
# # best result
# model = XGBClassifier(learning_rate=0.013,
#                       colsample_bytree=1,
#                       subsample=1,
#                       objective='binary:logistic',
#                       n_estimators=800,
#                       reg_alpha=0.25,
#                       max_depth=4,
#                       scale_pos_weight=250,
#                       gamma=0.3)
#
# pipe = Pipeline([('scaler', RobustScaler()),
#                  ('rf_classifier', model)])
#
# # evaluate train set
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# roc_auc_scores = cross_val_score(pipe, X_ov_train_stacked, y_ov_train_stacked, scoring='roc_auc', cv=cv, n_jobs=1)
# f1_scores = cross_val_score(pipe, X_ov_train_stacked, y_ov_train_stacked, scoring='f1', cv=cv, n_jobs=1)
# print('Mean ROC AUC: %.3f' % mean(roc_auc_scores))
# print('Mean F1: %.3f' % mean(f1_scores))
#
#
# # evaluate test set
# pipe.fit(X_ov_train_stacked, y_ov_train_stacked)
# pred_test = pipe.predict(X_ov_t_test)
# print(f1_score(y_ov_t_test, pred_test, average='binary', zero_division=1))
# print(roc_auc_score(y_ov_t_test, pred_test))
# conf_matrix = confusion_matrix(y_true=y_ov_t_test, y_pred=pred_test)
# print(conf_matrix.ravel())
#
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
# for i in range(conf_matrix.shape[0]):
#     for j in range(conf_matrix.shape[1]):
#         ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
#
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 5. save model as onnx
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# regist XGBoost for converting
# update_registered_converter(
#     XGBClassifier, 'XGBoostXGBClassifier',
#     calculate_linear_classifier_output_shapes, convert_xgboost,
#     options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
#
# model_onnx = convert_sklearn(pipe, 'pipeline_xgb', [('input', FloatTensorType([None, len(feature_list)]))])
#
# # add metadata
# meta = model_onnx.metadata_props.add()
# meta.key = "model_info"
# cfg = {"input_type": "float", "input_shape": [None, len(feature_list)], "labels": {0: "normal", 1: "fraud"}}
# meta.value = str(cfg)
#
# saved_model_path = ROOT_DIR + "/saved_models/td_test/"
# if not os.path.exists(saved_model_path):
#     os.makedirs(saved_model_path)
# with open(saved_model_path + "fd_xgboost_ov.onnx", "wb") as f:
#     f.write(model_onnx.SerializeToString())
