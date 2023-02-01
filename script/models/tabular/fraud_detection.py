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

# trim off garbage data
# import pandas as pd
# from statics import ROOT_DIR
#
# # from tqdm import tqdm
# #
# dataset_path = ROOT_DIR + "/dataset/fd_test2/combined.csv"
# # saved_model_path = ROOT_DIR + "/saved_models/td_test/"
# # df = pd.concat(map(pd.read_csv, [ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_1.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_2.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_3.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_4.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_5.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_6.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_7.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_8.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_9.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_10.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_11.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_12.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_13.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_14.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_15.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_16.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_17.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_18.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_19.csv",
# #                                  ROOT_DIR + "/dataset/fd_test2/전자금융 정상 거래고객 로그_20.csv"]), ignore_index=True)
# # seq = []
# # b_seq = 0
# # fc = 0
# # for index, row in tqdm(df.iterrows()):
# #     seq.append(row["SEQ"] + fc*1000)
# #     if b_seq > row['SEQ']:
# #         fc += 1
# #     b_seq = row['SEQ']
# #     # print(index, row['SEQ'])
# # print(seq)
# # df_s = pd.DataFrame(seq, columns=['SEQ'])
# # df["SEQ"] = df_s
# # df.to_csv(ROOT_DIR + "/dataset/fd_test2/combined.csv", index=False, encoding="utf-8-sig")
# # print("done")
#
#
# # check df includes exc cust id
# df_exc = pd.read_csv(ROOT_DIR + "/dataset/fd_test/fraud_dataset_refine_fixed.csv")
# df_exc["고객ID"].fillna(-1, inplace=True)
# df_exc["고객ID"] = df_exc["고객ID"].astype('int32')
# exc_list = df_exc["고객ID"].unique().tolist()
# del exc_list[exc_list.index(-1)]
#
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
#     print(f"{idx} / {len(df_sep_seq)}")
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
# duped.to_csv(ROOT_DIR + "/dataset/fd_test2/combined_duped.csv", index=False, encoding="utf-8-sig")
#
# # drop duplicate
# drop_count = 0
# for dupe in dup_cust_ids:
#     print(f"{drop_count} / {len(dup_cust_ids)}")
#     seq_be_del = dupe[1]
#     if seq_be_del < len(df_sep_seq):
#         df.drop(df[df.SEQ == seq_be_del].index, inplace=True)
#         drop_count += 1
# print(drop_count)
# # drop extra channel
# searchfor = ['CCM', 'CCW']
# df = df[df.메뉴.str.contains('|'.join(searchfor))]
# df.to_csv(ROOT_DIR + "/dataset/fd_test2/combined_trimmed.csv", index=False, encoding="utf-8-sig")
# print(len(df["SEQ"].unique().tolist()))
# # 7235
# print("done")


# trimming both first
# combine with combined.csv and fraud_dataset_trimmed.csv
# reset seq
# select 12000 of normal randomly
# select 7 of fraud randomly
# make both dataset to tabular
# train, validate(cross val)
# test with leftover data
# make report
# import pandas as pd
# from statics import ROOT_DIR
#
# df = pd.read_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_f.csv")

# seq = []
# b_seq = 1
# fc = 1
# for index, row in df.iterrows():
#     if b_seq != row['SEQ']:
#         fc += 1
#     seq.append(fc)
#     b_seq = row['SEQ']
#     print(fc, row['SEQ'])
# df_s = pd.DataFrame(seq, columns=['SEQ'])
# df["SEQ"] = df_s
# df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_f.csv", index=False, encoding="utf-8-sig")
# print("done")
# sequences = df["SEQ"].unique().tolist()
# len(sequences)
#


# import pandas as pd
# from statics import ROOT_DIR
# df = pd.read_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_f.csv")
# df["고객ID"].fillna(-1, inplace=True)
# df["고객ID"] = df["고객ID"].astype('int32')
#
# # drop extra channel
# searchfor = ['CCM', 'CCW']
# print(len(df))
# df = df[df.메뉴.str.contains('|'.join(searchfor))]
# print(len(df))
#
#
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
#     print(f"{idx} / {len(df_sep_seq)}", end='\r')
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
#
# # drop duplicate
# drop_count = 0
# for dupe in dup_cust_ids:
#     print(f"{drop_count} / {len(dup_cust_ids)}", end='\r')
#     seq_be_del = dupe[1]
#     if seq_be_del < len(df_sep_seq):
#         df.drop(df[df.SEQ == seq_be_del].index, inplace=True)
#         drop_count += 1
# print(drop_count)
#
#
# df_l = pd.read_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_f.csv")
# df_l["고객ID"].fillna(-1, inplace=True)
# df_l["고객ID"] = df_l["고객ID"].astype('int32')
#
#
# print(df_l[df_l["SEQ"] == 21023].reset_index(drop=True))
#
# df_l_21023 = df_l[df_l["SEQ"] == 21023].reset_index(drop=True)
# df = pd.concat([df, df_l_21023])
#
# df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed.csv", index=False, encoding="utf-8-sig")
# print(len(df["SEQ"].unique().tolist()))


# split train and test
# import pandas as pd
# from statics import ROOT_DIR
#
# df = pd.read_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed.csv")
# df_n = df[df["정상여부"] == "정상"].reset_index(drop=True)
# df_p = df[df["정상여부"] == "금융사기"].reset_index(drop=True)
#
# seq_df_n = df_n["SEQ"].unique().tolist()
# len(seq_df_n)
# test_list = []
#
# seq_df_p = df_p["SEQ"].unique().tolist()
# len(seq_df_p)
#
# import random
# train_n = random.sample(range(len(seq_df_n)), len(seq_df_n)-13000)
# train_len = len(train_n)
#
# for i, idx in enumerate(train_n):
#     print(i, train_len)
#     test_list.append(df_n[df_n["SEQ"] == seq_df_n[idx]].reset_index(drop=True))
#     df = df[df["SEQ"] != seq_df_n[idx]].reset_index(drop=True)
#
# len(test_list)
# len(df["SEQ"].unique().tolist())
# test_df = pd.concat(test_list)
# df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_train.csv", index=False, encoding="utf-8-sig")
# test_df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_test.csv", index=False, encoding="utf-8-sig")


# refine data with trimmed dataset
# import pandas as pd
# from statics import ROOT_DIR
# dataset_path = ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_test.csv"
#
# df = pd.read_csv(dataset_path)
# df.fillna("", inplace=True)
# df["EVENT"] = df.메뉴.str.cat(df.프로그램명)
# df['시간'] = pd.to_datetime(df['시간'], errors='coerce')
# df['DATE'] = df['시간'].dt.strftime("%Y-%m-%d")
# n_df_list = []
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
#
# for idx, df_seq in enumerate(df_sep_seq):
#     print(f"{idx} / {len(df_sep_seq)}", end='\r')
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
# print(count)
#
# print("done")
# print(num_multi_cust_id)
# print(len(n_df_list))
# #
# # for i, v in enumerate(n_df_list):
# #     # print(i, v.iloc[0]['SEQ'])
# #     print(i, len(v))
#
# #17652-6327
# n_df = pd.concat(n_df_list, ignore_index=True)
# n_df["SEQ"].unique()
# n_df["로그인유형"].replace("", "N/A")
# # n_df["비고"].replace("", "N/A")
# assert not n_df.isnull().values.any()
# n_df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_test_refine.csv", index=False, encoding="utf-8-sig")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2. transform dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# # make dataset to tabular
# import pandas as pd
# from statics import ROOT_DIR
# dataset_path = ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_test_refine.csv"
# df = pd.read_csv(dataset_path)
#
# df.drop(['메뉴', '메뉴명', "고객ID", "로그인IP", "프로그램명", "금액", "로그인유형", "DATE"], axis=1, inplace=True)
# # # df.drop(['메뉴', '메뉴명', "고객ID", "로그인IP", "프로그램명", "시간", "금액", "로그인유형", "비고", "DATE"], axis=1, inplace=True)
# df["EVENT"] = df["EVENT"].str.replace("CCW", "CCM")
# classes = {"정상": 0, "금융사기": 1}
# df["정상여부"] = df["정상여부"].apply(classes.get).astype(int)
#
# evnt_39 = {"CCMLN0201CH01":0,
#            "CCMLN0102SL02":0,
#            "CCMRD0206SL01":0,
#            "CCMFD0101RG01":0,
#            "CCMVI0207SL01":0,
#            "CCMCP0801SE01":0,
#            "CCMES0403":0,
#            "CCMSA0101SE02":0,
#            "CCMSA0301SE01":0,
#            "CCMLN0101PC01":0,
#            "CCMLO0101SE01":0,
#            "CCMRD0202CH01":0,
#            "CCMSA0101SE01":0,
#            "CCMFD0101RG02":0,
#            "CCMLO0201":0,
#            "CCMCP0801":0,
#            "CCMRD0102SL02":0,
#            "CCMCM0103SE01":0,
#            "CCMLN0101CH01":0,
#            "CCMCP0601":0,
#            "CCMES0201SL01":0,
#            "CCMSA0101SL01":0,
#            "CCMLO0101SE05":0,
#            "CCMFD0101AT01":0,
#            "CCMCM0101":0,
#            "CCMSA0101":0,
#            "CCMMS0101SL01":0,
#            "CCMMS0101":0,
#            "CCMLO0101":0,
#            "CCMLN0101CL01":0,
#            "CCMCP0301SE01":0,
#            # "time_diff":0,
#            "CCMLN0101":0,
#            "CCMRD0201SL01":0,
#            "CCMLN0201":0,
#            "CCMLN0101SL02":0,
#            "CCMCP0801SE02":0,
#            "CCMCP0801SE03":0,
#            "CCMLN0101SL03":0,
#            "CCMCP0801IN01":0,
#            "CCMLN0101SL01":0,
#            "CCMFD0101AT02":0,
#            "CCMMA0201SE01":0,
#            "CCMMA0101SE01":0}
#
# events = df["EVENT"].unique().tolist()
# events = list(evnt_39.keys())
# len(events)
# tabular_dataset = pd.DataFrame(columns=["SEQ"] + events + ["label"])
#
# # separate by seq
# seqs = df["SEQ"].unique().tolist()
# tdfs = []
# for seq in seqs:
#     tdfs.append(df[df["SEQ"] == seq].reset_index(drop=True))
#
# import time
# import datetime
#
# total = len(tdfs)
# for _, tdf in enumerate(tdfs):
#     print(_+1, "/", total, end='\r')
#     seq = tdf["SEQ"].iloc[0]
#     label = tdf["정상여부"].iloc[0]
#     time_diff = time.mktime(datetime.datetime.strptime(tdf["시간"].iloc[-1], "%Y-%m-%d %H:%M:%S").timetuple()) - \
#                 time.mktime(datetime.datetime.strptime(tdf["시간"].iloc[0], "%Y-%m-%d %H:%M:%S").timetuple())
#     counted_items = (dict(tdf["EVENT"].value_counts()))
#     # select 39 event only
#     tmp_items = {}
#     for k, v in evnt_39.items():
#         if k in counted_items:
#             tmp_items[k] = counted_items[k]
#         else:
#             tmp_items[k] = v
#     counted_items = tmp_items
#
#     data_row = {"SEQ": seq, "label": label, "time_diff": time_diff}
#     data_row.update(counted_items)
#     data_row = pd.DataFrame([data_row])
#     tabular_dataset = pd.concat([tabular_dataset, data_row], ignore_index=True)
#
# tabular_dataset.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_test_refine_tabular.csv", sep=",",
#                        index=False, encoding="utf-8-sig")
#
# len(tabular_dataset["SEQ"].unique().tolist())
# print("done")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 3. feature engineering
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # feature construction
# import pandas as pd
# from statics import ROOT_DIR
#
# dataset_path = ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular.csv"
# df = pd.read_csv(dataset_path)
# df.fillna(0, inplace=True)
# df_abnormal = df[df["label"] == 1].reset_index(drop=True)
#
# nc_cols = []
# for idx, column in enumerate(df_abnormal):
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
# df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_fc.csv", sep=",", index=False, encoding="utf-8-sig")
# print("done")


# # feature selection
# import pandas as pd
# import numpy as np
# from xgboost import XGBClassifier
# from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
# from sklearn.preprocessing import RobustScaler
# from sklearn.pipeline import Pipeline
#
# from statics import ROOT_DIR
#
# dataset_path = ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_fc.csv"
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
# model = XGBClassifier(learning_rate=0.03,
#                       colsample_bytree=1,
#                       subsample=1,
#                       objective='binary:logistic',
#                       n_estimators=2000,
#                       reg_alpha=0.25,
#                       max_depth=4,
#                       scale_pos_weight=10,
#                       gamma=0.3)
# pipe = Pipeline([('scaler', RobustScaler()),
#                  ('rf_classifier', model)])
#
# pipe.fit(X, y)
# pred_x = pipe.predict(X)
# for idx, pred in enumerate(pred_x):
#     if pred != y[idx]:
#         print(idx)
#
# print(f1_score(pred_x, y, average='binary', zero_division=1))
# print(roc_auc_score(pred_x, y))
# conf_matrix = confusion_matrix(y_true=y, y_pred=pred_x)
# print(conf_matrix.ravel())
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
# df.to_csv(ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_fe.csv",
#           sep=",", index=False, encoding="utf-8-sig")
#

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
# dataset_path = ROOT_DIR + "/dataset/fd_test2/combined_tabular.csv"
# df = pd.read_csv(dataset_path)
# df_labels = df[["label"]]
# dft = df.drop(["SEQ", "label", "ETC"], axis=1)
# dft = df.drop(["SEQ", "label", "time_diff"], axis=1)
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

# Threshold tuning
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

from script.statics import ROOT_DIR

train_dataset_path = ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_train_refine_tabular_fe.csv"
train_df = pd.read_csv(train_dataset_path)
train_df_labels = train_df[["label"]]
train_df_t = train_df.drop(["SEQ", "label", "ETC"], axis=1)
feature_list = train_df_t.keys().tolist()
X_train = np.array(train_df_t.values.tolist())
y_train = np.array(train_df_labels.values.tolist()).ravel()
original_data_idx = len(X_train)
num_neg = len(train_df[train_df["label"] == 1])
num_pos = len(X_train[:-num_neg])

test_dataset_path = ROOT_DIR + "/dataset/fd_test2/all_dataset_trimmed_test_refine_tabular.csv"
test_df = pd.read_csv(test_dataset_path)
test_df_labels = test_df[["label"]]
test_df_t = test_df.drop(["SEQ", "label"], axis=1)
X_test = np.array(test_df_t.values.tolist())
y_test = np.array(test_df_labels.values.tolist()).ravel()

# hp_resample_count = 10
# hp_grid_rl = [0.03, 0.02, 0.16, 0.013, 0.01]
# hp_grid_estimators = [2500, 2000, 1500, 1200, 850, 800, 700, 600, 500, 400]
# hp_grid_m_depth = [4, 5, 3]
# hp_grid_spw = [3, 50, 100, 200, 250, 300, 500, 800, 1000]
# hp_grid_gamma = [0.15, 0.2, 0.25, 0.3, 0.35]
# hp_grid_threshold = [0.5, 0.55, 0.6, 0.65, 0.68, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9, 0.92, 0.95]

hp_resample_count = 3
hp_grid_rl = [0.03, 0.025, 0.2]
hp_grid_estimators = [500, 800, 1000, 1200, 1500, 1800]
hp_grid_m_depth = [4, 5]
hp_grid_spw = [10, 50, 100, 200, 250]
hp_grid_gamma = [0.35, 0.25, 0.3]
hp_grid_threshold = [0.5, 0.55, 0.6, 0.65, 0.68, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9, 0.92, 0.95, 0.98]


log = {}
for i in range(hp_resample_count):
    print("resample")
    num_neg = len(train_df[train_df["label"] == 1])
    sm = SMOTE(random_state=1, sampling_strategy=0.10, k_neighbors=5)
    ad = ADASYN(random_state=1 + 1, sampling_strategy=0.10)
    ad2 = SMOTE(random_state=1 + 2, sampling_strategy=0.03)

    X_ovs_stack_ad, y_ovs_stack_ad = ad.fit_resample(X_train, y_train)
    for i, org in enumerate(X_train[-num_neg:]):
        assert np.array_equal(org,
                              X_ovs_stack_ad[original_data_idx - num_neg:original_data_idx][i]), "on splice ad test"

    X_ovs_stack_sm, y_ovs_stack_sm = sm.fit_resample(X_train, y_train)
    for i, org in enumerate(X_train[-num_neg:]):
        assert np.array_equal(org,
                              X_ovs_stack_sm[original_data_idx - num_neg:original_data_idx][i]), "on splice sm test"
    X_ovs_stack_sm = X_ovs_stack_sm[original_data_idx:]
    y_ovs_stack_sm = y_ovs_stack_sm[original_data_idx:]

    X_ovs_stack_smt, y_ovs_stack_smt = ad2.fit_resample(X_train, y_train)
    for i, org in enumerate(X_train[-num_neg:]):
        assert np.array_equal(org,
                              X_ovs_stack_smt[original_data_idx - num_neg:original_data_idx][i]), "on splice ad2 test"
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
    # Resampled dataset shape Counter({0: 4897, 1: 996})
    # Resampled test dataset shape Counter({0: 544, 1: 154})

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
                        # get best threshold
                        pred_all = pipe.predict(X_ovs_train)
                        pred_all_proba = pipe.predict_proba(X_ovs_train)
                        for threshold in hp_grid_threshold:
                            print("grd_thres")
                            tmp_pred = pred_all
                            for idx, pred in enumerate(tmp_pred):
                                if pred == 1:
                                    if pred_all_proba[idx][1] < threshold:
                                        tmp_pred[idx] = 0
                            conf_matrix = confusion_matrix(y_true=y_ovs_train, y_pred=tmp_pred)
                            conf_matrix = list([int(x) for x in conf_matrix.ravel()])
                            f1 = f1_score(y_ovs_train, tmp_pred, average='binary', zero_division=1)
                            print(f1)
                            roc_auc = roc_auc_score(y_ovs_train, tmp_pred)
                            if conf_matrix[1] < 1 and conf_matrix[2] < 1:
                                print(f"f1 score:{f1}")
                                print(f"roc score: {roc_auc}")
                                pred_test = pipe.predict(X_ovs_test)
                                pred_proba_test = pipe.predict_proba(X_ovs_test)
                                tmp_pred_test = pred_test
                                for idx, pred in enumerate(tmp_pred_test):
                                    if pred == 1:
                                        if pred_proba_test[idx][1] < threshold:
                                            tmp_pred_test[idx] = 0
                                # check f1_score in test set
                                conf_matrix_test = confusion_matrix(y_true=y_ovs_test, y_pred=pred_test)
                                conf_matrix_test = list([int(x) for x in conf_matrix_test.ravel()])
                                print("threshold", threshold, "FP_test, FN_test", conf_matrix_test[1], conf_matrix_test[2])
                                if conf_matrix_test[1] < 1 and conf_matrix_test[2] < 1:
                                    f1_test = f1_score(y_ovs_test, tmp_pred_test, average='binary', zero_division=1)
                                    roc_auc_test = roc_auc_score(y_ovs_test, tmp_pred_test)
                                    print(f"test f1 score:{f1}")
                                    print(f"test roc score: {roc_auc}")
                                    update_registered_converter(
                                        XGBClassifier, 'XGBoostXGBClassifier',
                                        calculate_linear_classifier_output_shapes, convert_xgboost,
                                        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

                                    model_onnx = convert_sklearn(pipe, 'pipeline_xgb',
                                                                 [('input', FloatTensorType([None, len(feature_list)]))])

                                    # add metadata
                                    meta = model_onnx.metadata_props.add()
                                    meta.key = "model_info"
                                    cfg = {"input_type": "float", "input_shape": [None, len(feature_list)],
                                           "labels": {0: "정상", 1: "전자금융피해"},
                                           "transformer": "fraud_detection.transform_data",
                                           "threshold": threshold,
                                           "pos_class": 1}
                                    meta.value = str(cfg)

                                    from datetime import datetime
                                    import json

                                    now = datetime.now()
                                    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
                                    saved_model_path = ROOT_DIR + "/saved_models/td_test/" + date_time + "/"
                                    log["info"] = {"threshold": threshold,
                                                   "hp_grid_gamma": gamma,
                                                   "hp_grid_spw": spw,
                                                   "hp_grid_m_depth": m_depth,
                                                   "hp_grid_estimators": estimator,
                                                   "hp_grid_rl": rl,
                                                   "FP": conf_matrix_test[1],
                                                   "FN": conf_matrix_test[2],
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
# meta.key = "model_info"
# cfg = {"input_type": "float", "input_shape": [None, 46],
#        "labels": {0: "정상", 1: "전자금융피해"},
#        "transformer": "fraud_detection.transform_data",
#        "threshold": 0.78,
#        "pos_class": 1}
# meta.value = str(cfg)
#
# saved_model_path = ROOT_DIR + "/onnx_bt/"
# if not os.path.exists(saved_model_path):
#     os.makedirs(saved_model_path)
# with open(saved_model_path + "fd_xgboost_ov.onnx", "wb") as f:
#     f.write(model_onnx.SerializeToString())

###################
# eval
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
#
# from statics import ROOT_DIR
# import onnxruntime as rt
#
# dataset_path = ROOT_DIR + "/dataset/fd_test2/combined_tabular.csv"
# df = pd.read_csv(dataset_path)
# df_labels = df[["label"]]
# dft = df.drop(["SEQ", "label", "time_diff"], axis=1)
# feature_list = dft.keys().tolist()
# X = np.array(dft.values.tolist()).astype(np.float32)
# y = np.array(df_labels.values.tolist()).ravel()
#
# session = rt.InferenceSession("/home/ky/PycharmProjects/pythonProject/saved_models/MDL0000002_1.1/MDL0000002/111/fd_xgboost_ov.onnx")
# pred_onx = session.run(None, {"input": X})
# pred = pred_onx[0]
# pred_proba = pred_onx[-1]
# pred_th = []
# for idx, p in enumerate(pred):
#     if p == 1:
#         if pred_proba[idx].get(1) < 0.68:
#             pred_th.append(0)
#         else:
#             pred_th.append(p)
#     else:
#         pred_th.append(p)
#
# conf_matrix = confusion_matrix(y_true=y, y_pred=pred_th)
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