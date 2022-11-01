# # make dataset
#
# import pandas as pd
# from statics import ROOT_DIR
#
# dataset_path = ROOT_DIR + "/dataset/fd_test/fd_dataset.csv"
# df = pd.read_csv(dataset_path)
# print(df.keys())
# df = df[["MID", "IP", "PID", "PR", "Class"]]
#
# # CCM -> CCW check it is good to go
# df["MID"] = df["MID"].str.replace("CCW", "CCM")
# df[["PID"]] = df[["PID"]].fillna(value="NA")
# df["EVT"] = df.MID.str.cat(df.PID, sep='-')
#
# # check null
# assert not df.isnull().values.any()
#
# # uniques to col
# events = df["EVT"].unique().tolist()
# len(events)
#
# # sep with ip
# ips = df["IP"].unique().tolist()
# tdfs = []
# for ip in ips:
#     tdfs.append(df[df['IP'] == ip].reset_index(drop=True))
#
# begin_event = "CCMLO0101"   # login
# dataset = pd.DataFrame(columns=["IP"] + events + ["label"])
#
# for tdf in tdfs:
#     ip = tdf["IP"].iloc[0]
#     label = tdf["Class"].iloc[0]
#     start_idxes = tdf.index[tdf["MID"] == begin_event].tolist()
#     end_idxes = tdf.index[tdf["PR"] != 0].tolist()
#     start_idx = start_idxes[0]
#     if len(start_idxes) == 0:
#         continue
#     if len(end_idxes) == 0:
#         temp_df = tdf[start_idx:]
#         counted_items = (dict(temp_df["EVT"].value_counts()))
#         data_row = {"IP": ip, "label": label}
#         data_row.update(counted_items)
#         data_row = pd.DataFrame([data_row])
#         dataset = pd.concat([dataset, data_row], ignore_index=True)
#     else:
#         for end_idx in end_idxes:
#             temp_df = tdf[start_idx:end_idx+1]
#             counted_items = (dict(temp_df["EVT"].value_counts()))
#             data_row = {"IP": ip, "label": label}
#             data_row.update(counted_items)
#             data_row = pd.DataFrame([data_row])
#             dataset = pd.concat([dataset, data_row], ignore_index=True)
#             for it_start_idx in start_idxes:
#                 if it_start_idx > end_idx+1:
#                     start_idx = it_start_idx
#
# dataset.to_csv(ROOT_DIR + "/dataset/fd_test/fd_dataset_tabular.csv", na_rep=0, sep=",", index=False)


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
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

n_features = 50
X, y = make_classification(n_classes=2, class_sep=3, weights=[0.01, 0.99], n_informative=2, n_redundant=1, flip_y=0,
                           n_features=n_features, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape %s' % Counter(y))


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
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



# don't oversampling with smote to much
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X, y)
# print('Resampled dataset shape %s' % Counter(y_res))
#
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
#
# model = TSNE(n_components=2, perplexity=5, n_iter=3000, learning_rate="auto", init="pca")
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


# XGboost with some of real data with oversampled data
# under sampling tree with some of real data with oversampled data

# get score and check with real data both
