# # make dataset
#
# data trimming
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

# drop duplicate
#
# for dupe in dup_cust_ids:
#     seq_be_del = dupe[1]
#     print(seq_be_del)
#     if seq_be_del < 1000:
#         df.drop(df[df.SEQ == seq_be_del].index, inplace=True)
#
# searchfor = ['CCM', 'CCW']
# df = df[df.메뉴.str.contains('|'.join(searchfor))]
# df.to_csv(ROOT_DIR + "/dataset/fd_test/fd_dataset_trimmed.csv", index=False)


# refine data with trimmed dataset
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
# start_events = ["CCMLO0101", "CCWLO0101", "CCMMS0101SL01", "CCWMS0101SL01"]
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
# n_df.to_csv(ROOT_DIR + "/dataset/fd_test/fraud_dataset_refine.csv", index=False, encoding="utf-8-sig")
# n_df.keys()


# make dataset to tabular
# import pandas as pd
# from statics import ROOT_DIR
# dataset_path = ROOT_DIR + "/dataset/fd_test/fraud_dataset_refine.csv"
# df = pd.read_csv(dataset_path)
# df.drop(['메뉴', '메뉴명', "고객ID", "로그인IP", "프로그램명", "시간", "금액", "로그인유형", "비고", "DATE"], axis=1, inplace=True)
# df["EVENT"] = df["EVENT"].str.replace("CCW", "CCM")
# classes = {"정상": 0, "금융사기": 1}
# df["정상여부"] = df["정상여부"].apply(classes.get).astype(int)
#
# events = df["EVENT"].unique().tolist()
# tabular_dataset = pd.DataFrame(columns=["SEQ"] + events + ["label"])
#
# # sep with seq
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
# tabular_dataset.to_csv(ROOT_DIR + "/dataset/fd_test/fraud_dataset_tabular.csv", sep=",",
#                        index=False, encoding="utf-8-sig")
#
# print("done")


# feature engineering




# make model

# import pandas as pd
# import numpy as np
# from statics import ROOT_DIR
#
# dataset_path = ROOT_DIR + "/dataset/fd_test/fd_dataset_tabular.csv"
# df = pd.read_csv(dataset_path)
# df_labels = df[["label"]]
# df.drop(["label", "IP"], axis=1, inplace=True)
#
#
# def convert_to_float(collection):
#     floats = [float(el) for el in collection]
#     return np.array(floats)
#
# df_numeric = pd.concat([df.apply(convert_to_float)], axis=1)
# df_numeric = df_numeric.values.tolist()
# df_numeric = np.array(df_numeric)
#
#
# # projection 2d
# from sklearn.manifold import TSNE
#
# model = TSNE(n_components=2, perplexity=1, n_iter=3000, learning_rate=200, init="pca")
# res = model.fit_transform(df_numeric)
#
# import matplotlib.pyplot as plt
#
# plt.scatter(res[:, 0], res[:, 1])
# plt.show()

# smote
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn

# use ADASYN for additional data, smote will be better
# use XGBoost
# use PCA or LDA
# use t-sne to visualization
# use under bagging random forest

# build logics with sample dataset
# from collections import Counter
# from sklearn.datasets import make_classification
# from imblearn.over_sampling import SMOTE

# n_features = 50
# X, y = make_classification(n_classes=2, class_sep=3, weights=[0.01, 0.99], n_informative=2, n_redundant=1, flip_y=0,
#                            n_features=n_features, n_clusters_per_class=1, n_samples=1000, random_state=10)
# print('Original dataset shape %s' % Counter(y))
#
#
# from sklearn.preprocessing import StandardScaler
# X = StandardScaler().fit_transform(X)
#
# columns = [str(x) for x in range(n_features)]
# df = pd.DataFrame(X, columns=columns)
# df["labels"] = y
#
# df_label_0 = df[df["labels"] == 0].reset_index(drop=True)
# df_label_1 = df[df["labels"] == 1].reset_index(drop=True)
#
# df_X_label_0 = df_label_0.iloc[:, :-1]
# list_X_label_0 = df_X_label_0.values.tolist()
#
# df_X_label_1 = df_label_1.iloc[:, :-1]
# list_X_label_1 = df_X_label_1.values.tolist()
#
# # visualization
# # use t-sne
# model = TSNE(n_components=2, perplexity=1, n_iter=3000, learning_rate="auto", init="pca")
# res_0 = model.fit_transform(np.array(list_X_label_0))
#
# model = TSNE(n_components=2, perplexity=5, n_iter=3000, learning_rate="auto", init="pca")
# res_1 = model.fit_transform(np.array(list_X_label_1))
#
# import matplotlib.pyplot as plt
#
# plt.scatter(res_0[:, 0], res_0[:, 1])
# plt.scatter(res_1[:, 0], res_1[:, 1])
# plt.show()

# use pca -> t-sne is better
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
# res_0 = pca.fit_transform(np.array(list_X_label_0))
# res_1 = pca.fit_transform(np.array(list_X_label_1))
#
# plt.scatter(res_0[:, 0], res_0[:, 1])
# plt.scatter(res_1[:, 0], res_1[:, 1])
# plt.show()


# oversampling all
# don't oversampling with smote to much
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X, y)
# print('Resampled dataset shape %s' % Counter(y_res))

# columns = [str(x) for x in range(n_features)]
# df = pd.DataFrame(X_res, columns=columns)
# df["labels"] = y_res
#
# df_label_0 = df[df["labels"] == 0].reset_index(drop=True)
# df_label_1 = df[df["labels"] == 1].reset_index(drop=True)
#
# df_X_label_0 = df_label_0.iloc[:, :-1]
# list_X_label_0 = df_X_label_0.values.tolist()
#
# df_X_label_1 = df_label_1.iloc[:, :-1]
# list_X_label_1 = df_X_label_1.values.tolist()

# model = TSNE(n_components=2, perplexity=5, n_iter=3000, learning_rate="auto", init="pca")
# x_res_0 = model.fit_transform(np.array(list_X_label_0))
#
# model = TSNE(n_components=2, perplexity=5, n_iter=3000, learning_rate="auto", init="pca")
# x_res_1 = model.fit_transform(np.array(list_X_label_1))
#
# import matplotlib.pyplot as plt
#
# plt.scatter(res_0[:, 0], res_0[:, 1])
# plt.scatter(res_1[:, 0], res_1[:, 1])
# plt.show()

# random forest with smote
# from numpy import mean
# from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
#
# # define model
# model = RandomForestClassifier(n_estimators=10)
# # define evaluation procedure
# pipe = Pipeline([('scaler', StandardScaler()),
#                  ('rf_classifier', model)])
#
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# # evaluate model
# cv_result = cross_validate(pipe, X_res, y_res, scoring='roc_auc', cv=cv, n_jobs=1, return_estimator=True,
#                            return_train_score=True)
# # print(cv_result["estimator"])
# # print(cv_result["train_score"])
# # print(cv_result["test_score"])
# # summarize performance
# print('Mean Train ROC AUC: %.3f' % mean(cv_result["train_score"]))
# est_0 = cv_result["estimator"][0]
# est_0.score(X, y)
# # export to onnx
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
# model_onnx = convert_sklearn(est_0, 'pipeline_estimator',
#                 [('input', FloatTensorType([None, n_features]))])
#
# import os
# if not os.path.exists(saved_model_path):
#     os.makedirs(saved_model_path)
#
# with open(saved_model_path + "ft_test_random_frest.onnx", "wb") as f:
#     f.write(model_onnx.SerializeToString())
#
# import onnxruntime as rt    # pip install onnxruntime
# print("predict", est_0.predict(X[:5]))
# print("predict_proba", est_0.predict_proba(X[:5]))
#
# sess = rt.InferenceSession(saved_model_path + "ft_test_random_frest.onnx")
# pred_onx = sess.run(None, {"input": X[:5].astype(np.float32)})
# print("predict", pred_onx[0])
# print("predict_proba", pred_onx[1][:1])
#

# XGBoost with smote
# from numpy import mean
# from sklearn.datasets import make_classification
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from xgboost import XGBClassifier   #pip install xgboost
# from sklearn.model_selection import GridSearchCV
# from skl2onnx.common.data_types import FloatTensorType
#
# model = XGBClassifier()
# weights = [1, 10, 25, 50, 75, 99, 100, 1000]
# param_grid = dict(scale_pos_weight=weights)
# # define evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=cv, scoring='roc_auc')
# # execute the grid search
# grid_result = grid.fit(X, y)
# # report the best configuration
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# # report all configurations
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# from sklearn.pipeline import Pipeline
# pipe = Pipeline([('scaler', StandardScaler()),
#                  ('XGB_Classifier', model)])
# pipe.fit(X, y)
#
# from skl2onnx import convert_sklearn, update_registered_converter
# from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
# from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost # pip install onnxmltools
#
# update_registered_converter(
#     XGBClassifier, 'XGBoostXGBClassifier',
#     calculate_linear_classifier_output_shapes, convert_xgboost,
#     options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
#
# model_onnx = convert_sklearn(pipe, 'pipeline_xgb',
#                 [('input', FloatTensorType([None, n_features]))])
#
# import os
# if not os.path.exists(saved_model_path):
#     os.makedirs(saved_model_path)
#
# with open(saved_model_path + "ft_test_xgboost.onnx", "wb") as f:
#     f.write(model_onnx.SerializeToString())
#
# print("predict", pipe.predict(X[:5]))
# print("predict_proba", pipe.predict_proba(X[:1]))
#
# import onnxruntime as rt
# sess = rt.InferenceSession(saved_model_path + "ft_test_xgboost.onnx")
# pred_onx = sess.run(None, {"input": X[:5].astype(np.float32)})
# print("predict", pred_onx[0])
# print("predict_proba", pred_onx[1][:1])

# save with metadata
# import onnx
# model = onnx.load(saved_model_path + "ft_test_xgboost.onnx")
# meta = model.metadata_props.add()
# meta.key = "cfg"
# cfg = {"input_type": "float", "input_shape": [None, n_features], "labels": {0: "fraud", 1: "normal"}}
# meta.value = str(cfg)
# onnx.save(model, saved_model_path + "ft_test_xgboost_1.onnx")
#
# # loading
# cfg_s = eval(onnx.load(saved_model_path + "ft_test_xgboost_1.onnx").metadata_props[0].value)
# print(cfg_s)

# evaluate model
# scores = cross_val_score(model, X_res, y_res, scoring='roc_auc', cv=cv, n_jobs=1)
# summarize performance
# print('Mean ROC AUC: %.5f' % mean(scores))


# under bagging with imb data

# weighted XGBoost with imb data

# XGBoost model to onnx model -> save

# make deploy logic
