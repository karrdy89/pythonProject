# check default model
import tensorflow as tf

def multihead_attention(queries, keys, attention_dim: int, num_heads: int, dropout_rate: float):
    # Linear projections
    Q = tf.keras.layers.Dense(attention_dim, activation=None)(queries)  # (N, T_q, C)
    K = tf.keras.layers.Dense(attention_dim, activation=None)(keys)  # (N, T_k, C)
    V = tf.keras.layers.Dense(attention_dim, activation=None)(keys)  # (N, T_k, C)
    # --- MULTI HEAD ---
    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    # --- SCALED DOT PRODUCT ---
    # Multiplication
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
    # Key Masking
    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
    # Future blinding (Causality)
    diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
    masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
    paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
    # Query Masking
    query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (N, T_q, C)
    # Dropouts
    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    # Weighted sum
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
    # --- MULTI HEAD ---
    # concat heads
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
    # Residual connection
    outputs += queries
    return outputs


def point_wise_feed_forward(input_seq, dropout_rate: float, conv_dims: list):
    output = tf.keras.layers.Conv1D(filters=conv_dims[0], kernel_size=1, activation='relu', use_bias=True)(input_seq)
    output = tf.keras.layers.Dropout(dropout_rate)(output)
    output = tf.keras.layers.Conv1D(filters=conv_dims[1], kernel_size=1, activation=None, use_bias=True)(output)
    output = tf.keras.layers.Dropout(dropout_rate)(output)
    output += input_seq
    return output


def embedding(input_seq, item_embedding_layer, positional_embedding_layer):
    seq_embeddings = item_embedding_layer(input_seq)
    positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
    positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
    positional_embeddings = positional_embedding_layer(positional_seq)
    return seq_embeddings, positional_embeddings


def layer_normalization(input_seq, epsilon: float = 1e-8):
    inputs_shape = input_seq.get_shape()
    params_shape = inputs_shape[-1:]
    mean, variance = tf.nn.moments(input_seq, [-1], keepdims=True)
    beta = tf.zeros(params_shape)
    gamma = tf.ones(params_shape)
    normalized = (input_seq - mean) / ((variance + epsilon) ** .5)
    output = gamma * normalized + beta
    return output


def loss_function(y_true, y_pred):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)


from tensorflow.keras import layers
def get_model(vocab_size: int, vocabulary, max_len: int, num_labels: int, embedding_dim: int = 8,
              dropout_rate: float = 0.2, num_blocks: int = 1, attention_num_heads: int = 4, l2_reg: float = 1e-6,
              epsilon: float = 1e-8, learning_rate: float = 0.02, mask_token: str = ""):
    attention_dim = embedding_dim
    conv_dims = [embedding_dim, embedding_dim]

    input_meta = {"max_len": max_len, "transformer": "nbo.transform_data_m2"}
    inputs = layers.Input(shape=(None,), name='input', dtype=object)
    encoding_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocabulary,
                                                                             mask_token=mask_token)
    encoded_inputs = encoding_layer(inputs)
    positional_embedding_layer = tf.keras.layers.Embedding(max_len, embedding_dim,
                                                           name='positional_embeddings', mask_zero=True,
                                                           embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
    item_embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                     name='item_embeddings', mask_zero=True,
                                                     embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
    seq_embeddings, positional_embeddings = embedding(encoded_inputs, item_embedding_layer, positional_embedding_layer)
    seq_embeddings += positional_embeddings
    mask = tf.expand_dims(tf.cast(tf.not_equal(encoded_inputs, 0), tf.float32), -1)
    seq_embeddings = tf.keras.layers.Dropout(dropout_rate)(seq_embeddings)
    seq_embeddings *= mask
    seq_attention = seq_embeddings
    for i in range(num_blocks):
        seq_attention = multihead_attention(queries=layer_normalization(seq_attention, epsilon=epsilon),
                                            keys=seq_attention,
                                            attention_dim=attention_dim,
                                            num_heads=attention_num_heads,
                                            dropout_rate=dropout_rate)
        seq_attention = point_wise_feed_forward(seq_attention, dropout_rate=dropout_rate, conv_dims=conv_dims)
        seq_attention *= mask
    x = layer_normalization(seq_attention)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(10, name='fc_last', activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_labels, activation="softmax")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                         amsgrad=False)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    return model


import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2022-09-01_t.csv"
TESTSET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2023-05-20_t.csv"
df = pd.read_csv(DATASET_PATH)
test_df = pd.read_csv(TESTSET_PATH)

labels = df["label"].unique().tolist()
labels_dict = {}
class_weight = {}
for idx, label in enumerate(labels):
    label_df = df.loc[df['label'] == label]
    class_weight[idx] = len(label_df)
    labels_dict[label] = idx

lowest_class = min(class_weight, key=class_weight.get)
class_total = 0
for k, v in class_weight.items():
    class_total += v

for k, v in class_weight.items():
    if k == lowest_class:
        class_weight[k] = (1 / v) * (class_total / 2.0)
    else:
        class_weight[k] = (1 / v) * (class_total / 2.0)

df = df.replace({"label": labels_dict})
test_df = test_df.replace({"label": labels_dict})

y = df[["label"]]
y = y.values.tolist()
y_test = test_df[["label"]]
y_test = y_test.values.tolist()
df.drop(["label", "key"], axis=1, inplace=True)
test_df.drop(["label", "key"], axis=1, inplace=True)
X = df.stack().groupby(level=0).apply(list).tolist()
X_test = test_df.stack().groupby(level=0).apply(list).tolist()
X_tmp = []
y_tmp = []
MAX_LEN = 50
MIN_LEN = 1
PAD_VALUE = ""
for idx, x in enumerate(X):
    x_len = len(x)
    if x_len >= 1:
        if x_len >= MAX_LEN:
            X_tmp.append(x[-MAX_LEN:])
            y_tmp.append(y[idx])
        else:
            pad_size = MAX_LEN - x_len
            X_tmp.append([*x, *[PAD_VALUE] * pad_size])
            y_tmp.append(y[idx])
X = X_tmp
y = y_tmp

X_tmp = []
y_tmp = []
MAX_LEN = 50
MIN_LEN = 1
PAD_VALUE = ""
for idx, x in enumerate(X_test):
    x_len = len(x)
    if x_len >= 1:
        if x_len >= MAX_LEN:
            X_tmp.append(x[-MAX_LEN:])
            y_tmp.append(y_test[idx])
        else:
            pad_size = MAX_LEN - x_len
            X_tmp.append([*x, *[PAD_VALUE] * pad_size])
            y_tmp.append(y_test[idx])
X_test = X_tmp
y_test = y_tmp

# split train, valid, test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

# train
from itertools import chain
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np

max_len = MAX_LEN + 1
vocabs = list(set(chain(*X)))
mask_token = vocabs.pop(vocabs.index(""))
le = preprocessing.StringLookup()
le.adapt(vocabs)

vocab_size = len(le.get_vocabulary()) + 1
num_labels = len(labels)
model = get_model(vocab_size=vocab_size, vocabulary=vocabs, max_len=max_len, num_labels=num_labels,
                  mask_token=mask_token)

X_train = np.array(X_train, dtype='object')
y_train = np.array(y_train).astype(np.int)

X_valid = np.array(X_val, dtype='object')
y_valid = np.array(y_val).astype(np.int)

X_test = np.array(X_test, dtype='object')
y_test = np.array(y_test).astype(np.int)
y_test = y_test.ravel()


hist = model.fit(X_train, y_train,
                 validation_data=(X_valid, y_valid),
                 epochs=1,
                 batch_size=64,
                 verbose=1,
                 class_weight=class_weight)

pred = model.predict(X_test)
pred_class = pred.argmax(axis=-1).tolist()

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
f1 = f1_score(pred_class, y_test, average=None, zero_division=0)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=pred_class)
acc = accuracy_score(y_true=y_test, y_pred=pred_class)
prec = precision_score(y_true=y_test, y_pred=pred_class, average=None, zero_division=0)
recl = recall_score(y_true=y_test, y_pred=pred_class, average=None, zero_division=0)

unique, counts = np.unique(y_test, return_counts=True)
req_res = dict(zip(unique, counts))

REQ_DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/actual.txt"
req_ds = open(REQ_DATASET_PATH, 'r', encoding='utf-16')
Lines = req_ds.readlines()
req_data = []
for idx, line in enumerate(Lines):
    line = line.rstrip("\n")
    if idx == 0:
        continue
    line = eval(line)
    if not isinstance(line, list):
        line = line["INPUT"]
    line = list(reversed(line))
    if len(line) >= MAX_LEN:
        line = line[-MAX_LEN:]
    else:
        pad_size = MAX_LEN - len(line)
        line = [*line, *[PAD_VALUE] * pad_size]
    req_data.append(line)

req_data = np.array(req_data, dtype='object')

pred_req = model.predict(req_data)
pred_req_class = pred.argmax(axis=-1).tolist()
unique, counts = np.unique(pred_req_class, return_counts=True)
req_res = dict(zip(unique, counts))
req_res_t = sum(req_res.values())
res_ratio = {key: (value / req_res_t) for key, value in req_res.items()}



########################################################################################################################
# outlier detection
import pandas as pd
from collections import Counter
from itertools import chain
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2022-09-01~2023-04-14.csv"
df = pd.read_csv(DATASET_PATH)
labels = df["label"].unique().tolist()

train_target_df = df.loc[df['label'] != "UNK"].copy()
train_unk_df = df.loc[df['label'] == "UNK"].copy()

train_target_df.drop(["label", "key"], axis=1, inplace=True)
train_unk_df.drop(["label", "key"], axis=1, inplace=True)

train_target_X = train_target_df.stack().groupby(level=0).apply(list).tolist()
train_unk_X = train_unk_df.stack().groupby(level=0).apply(list).tolist()

X_target = []
MAX_LEN = 20
MIN_LEN = 1
for idx, x in enumerate(train_target_X):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_target.append(x[-MAX_LEN:])
        else:
            X_target.append(x)
vocabs = list(set(chain(*X_target)))
print(f"vocab size: {len(vocabs)} ")

base_feature = {}
for evt in vocabs:
    base_feature[evt] = 0
base_feature["etc"] = 0

X_target_tabular = []
for x in X_target:
    tmp_feature = base_feature.copy()
    counted_item = Counter(x)
    tmp_feature.update(dict(counted_item))
    X_target_tabular.append(tmp_feature)

X_target_tabular = pd.DataFrame(X_target_tabular)
X_target_tabular = X_target_tabular.stack().groupby(level=0).apply(list).tolist()
y_target_tabular = [0] * len(X_target_tabular)

X_unk = []
for idx, x in enumerate(train_unk_X):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_unk.append(x[-MAX_LEN:])
        else:
            X_unk.append(x)

X_unk_tabular = []
for x in X_unk:
    tmp_feature = base_feature.copy()
    for item in x:
        if item in vocabs:
            tmp_feature[item] += 1
        else:
            tmp_feature["etc"] += 1
    X_unk_tabular.append(tmp_feature)

X_unk_tabular = pd.DataFrame(X_unk_tabular)
X_unk_tabular = X_unk_tabular.stack().groupby(level=0).apply(list).tolist()
y_unk_tabular = [1] * len(X_unk_tabular)


X = X_target_tabular + X_unk_tabular
y = y_target_tabular + y_unk_tabular
X, y = shuffle(X, y)

model = XGBClassifier(learning_rate=0.035,
                      colsample_bytree=1,
                      subsample=1,
                      objective='binary:logistic',
                      n_estimators=2000,
                      reg_alpha=0.25,
                      max_depth=7,
                      scale_pos_weight=0.8,
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

important_features = []
print("feature ranking")
for i in range(len(importance)):
    print("{}. feature {} ({:.3f}) ".format(i+1, list(base_feature.keys())[indices[i]], importance[indices[i]]))
    if importance[indices[i]] > 0.002:
        important_features.append(list(base_feature.keys())[indices[i]])

len(important_features)
import pickle
# with open("./nv_important_features.pkl", "wb") as fp:
#     pickle.dump(important_features, fp)

import pickle
with open('./nv_important_features.pkl', 'rb') as f:
    important_features = pickle.load(f)

base_i_features = {}
for feature in important_features:
    base_i_features[feature] = 0

X_target_tabular_fi = []
for x in X_target:
    tmp_feature = base_i_features.copy()
    for item in x:
        if item in important_features:
            tmp_feature[item] += 1
    X_target_tabular_fi.append(tmp_feature)

X_target_tabular_fi = pd.DataFrame(X_target_tabular_fi)
X_target_tabular_fi = X_target_tabular_fi.stack().groupby(level=0).apply(list).tolist()

X_unk_tabular_fi = []
for x in X_unk:
    tmp_feature = base_i_features.copy()
    for item in x:
        if item in important_features:
            tmp_feature[item] += 1
    X_unk_tabular_fi.append(tmp_feature)

X_unk_tabular_fi = pd.DataFrame(X_unk_tabular_fi)
X_unk_tabular_fi = X_unk_tabular_fi.stack().groupby(level=0).apply(list).tolist()



from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

scaler = RobustScaler()
scaler.fit(X_target_tabular_fi)
X_train = scaler.transform(X_target_tabular_fi)
pca = PCA(n_components=20, whiten=True)
pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)

X_test = scaler.transform(X_unk_tabular_fi)
X_test = pca.transform(X_test)

X_train = X_train.tolist()
X_test = X_test.tolist()

# X_train = X_target_tabular_fi
# X_test = X_unk_tabular_fi
#
# SPLIT_RATIO = 0.5
# X_val_neg = X_train[int(len(X_train)*SPLIT_RATIO):]
# X_train = X_train[:int(len(X_train)*SPLIT_RATIO)]
y_train = [1] * len(X_train)
#
# y_val_neg = [1] * len(X_val_neg)
# # y_train = [1] * len(X_train)
#
# X_val_pos = X_test[int(len(X_test)*SPLIT_RATIO):]
# y_val_pos = [0] * len(X_val_pos)
# X_test = X_test[:int(len(X_test)*SPLIT_RATIO)]
# y_test = [0] * len(X_test)
#
y_test = [-1] * len(X_test)
#
# X_val = X_val_neg.tolist() + X_val_pos.tolist()
# # X_val = X_val_neg + X_val_pos
# y_val = y_val_neg + y_val_pos
# X_val, y_val = shuffle(X_val, y_val)
# #
# from sklearn.mixture import GaussianMixture
# from sklearn.isotonic import IsotonicRegression
# gmm_clf = GaussianMixture(covariance_type='diag', n_components=3, max_iter=1000)
# gmm_clf.fit(X_train)
# log_probs_val = gmm_clf.score_samples(X_val)
# isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
# isotonic_regressor.fit(log_probs_val, y_val)
#
# # Obtaining results on the test set
# log_probs_test = gmm_clf.score_samples(X_test)
# test_probabilities = isotonic_regressor.predict(log_probs_test)
# test_predictions = [1 if prob > 0.5 else 0 for prob in test_probabilities]
# unique, counts = np.unique(test_predictions, return_counts=True)
# gmm_res_t = dict(zip(unique, counts))
#
# log_probs_test = gmm_clf.score_samples(X_train.tolist() + X_test.tolist())
# test_probabilities = isotonic_regressor.predict(log_probs_test)
# test_predictions = [1 if prob > 0.5 else 0 for prob in test_probabilities]
# unique, counts = np.unique(test_predictions, return_counts=True)
# gmm_res = dict(zip(unique, counts))
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# f1 = f1_score(test_predictions, y_train + y_test, average=None, zero_division=1)
# conf_matrix = confusion_matrix(y_true=y_train + y_test, y_pred=test_predictions)
# acc = accuracy_score(y_true=y_train + y_test, y_pred=test_predictions)
# prec = precision_score(y_true=y_train + y_test, y_pred=test_predictions, average=None)
# recl = recall_score(y_true=y_train + y_test, y_pred=test_predictions, average=None)
#
if_clf = IsolationForest(contamination=0.1, max_features=3, max_samples='auto', n_estimators=200)
if_clf.fit(X_train)
#
# # if_pred = if_clf.predict(X_test)
# # unique, counts = np.unique(if_pred, return_counts=True)
# # if_clf_res_t = dict(zip(unique, counts))
#
if_pred = if_clf.predict(X_train + X_test)
unique, counts = np.unique(if_pred, return_counts=True)
if_clf_res = dict(zip(unique, counts))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
f1 = f1_score(if_pred, y_train + y_test, average=None, zero_division=1)
conf_matrix = confusion_matrix(y_true=y_train + y_test, y_pred=if_pred)
acc = accuracy_score(y_true=y_train + y_test, y_pred=if_pred)
prec = precision_score(y_true=y_train + y_test, y_pred=if_pred, average=None, zero_division=1)
recl = recall_score(y_true=y_train + y_test, y_pred=if_pred, average=None, zero_division=1)
#
#
# oc_svm_clf = svm.OneClassSVM(gamma=10, kernel='rbf', nu=1)
# oc_svm_clf.fit(X_train)
#
# svm_pred = oc_svm_clf.predict(X_train)
# unique, counts = np.unique(svm_pred, return_counts=True)
# svm_clf_res = dict(zip(unique, counts))
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# f1 = f1_score(svm_pred, y_train + y_test, average=None, zero_division=1)
# conf_matrix = confusion_matrix(y_true=y_train + y_test, y_pred=svm_pred)
# acc = accuracy_score(y_true=y_train + y_test, y_pred=svm_pred)
# prec = precision_score(y_true=y_train + y_test, y_pred=svm_pred, average=None, zero_division=1)
# recl = recall_score(y_true=y_train + y_test, y_pred=svm_pred, average=None, zero_division=1)

#
# LOF = LocalOutlierFactor(n_neighbors=15, novelty=True, contamination=0.25)
# LOF.fit(X_train)
#
# lof_pred = LOF.predict(X_train+X_test)
# unique, counts = np.unique(lof_pred, return_counts=True)
# los_res = dict(zip(unique, counts))
#
#
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# f1 = f1_score(lof_pred, y_train + y_test, average=None, zero_division=1)
# conf_matrix = confusion_matrix(y_true=y_train + y_test, y_pred=lof_pred)
# acc = accuracy_score(y_true=y_train + y_test, y_pred=lof_pred)
# prec = precision_score(y_true=y_train + y_test, y_pred=lof_pred, average=None, zero_division=1)
# recl = recall_score(y_true=y_train + y_test, y_pred=lof_pred, average=None, zero_division=1)
#
#
# if_pred = if_clf.predict(X_test)
# unique, counts = np.unique(if_pred, return_counts=True)
# if_clf_res_t = dict(zip(unique, counts))
#
# if_pred = if_clf.predict(X_train)
# unique, counts = np.unique(if_pred, return_counts=True)
# if_clf_res = dict(zip(unique, counts))
#
# svm_pred = oc_svm_clf.predict(X_test)
# unique, counts = np.unique(svm_pred, return_counts=True)
# svm_clf_res_t = dict(zip(unique, counts))
#
# svm_pred = oc_svm_clf.predict(X_train)
# unique, counts = np.unique(svm_pred, return_counts=True)
# svm_clf_res = dict(zip(unique, counts))
#
# lof_pred_t = LOF.predict(X_test)
# unique, counts = np.unique(lof_pred_t, return_counts=True)
# los_res_t = dict(zip(unique, counts))
#
# lof_pred = LOF.predict(X_train)
# unique, counts = np.unique(lof_pred, return_counts=True)
# los_res = dict(zip(unique, counts))


########################################################################################################################
# XGB novelty detection
import pickle
from itertools import chain
import os
from datetime import datetime
import json

import joblib
import pandas as pd
from collections import Counter
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType

# make dataset for get feature importance (train)
DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2022-09-01~2023-04-14.csv"
df = pd.read_csv(DATASET_PATH)
labels = df["label"].unique().tolist()

train_target_df = df.loc[df['label'] != "UNK"].copy()
train_unk_df = df.loc[df['label'] == "UNK"].copy()

train_target_df.drop(["label", "key"], axis=1, inplace=True)
train_unk_df.drop(["label", "key"], axis=1, inplace=True)

train_target_X = train_target_df.stack().groupby(level=0).apply(list).tolist()
train_unk_X = train_unk_df.stack().groupby(level=0).apply(list).tolist()

X_target = []
MAX_LEN = 20
MIN_LEN = 1
for idx, x in enumerate(train_target_X):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_target.append(x[-MAX_LEN:])
        else:
            X_target.append(x)
vocabs = list(set(chain(*X_target)))
print(f"vocab size: {len(vocabs)} ")

base_feature = {}
for evt in vocabs:
    base_feature[evt] = 0

X_target_tabular = []
for x in X_target:
    tmp_feature = base_feature.copy()
    counted_item = Counter(x)
    tmp_feature.update(dict(counted_item))
    X_target_tabular.append(tmp_feature)

X_target_tabular = pd.DataFrame(X_target_tabular)
X_target_tabular = X_target_tabular.stack().groupby(level=0).apply(list).tolist()
y_target_tabular = [0] * len(X_target_tabular)

X_unk = []
for idx, x in enumerate(train_unk_X):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_unk.append(x[-MAX_LEN:])
        else:
            X_unk.append(x)

X_unk_tabular = []
for x in X_unk:
    tmp_feature = base_feature.copy()
    for item in x:
        if item in vocabs:
            tmp_feature[item] += 1
    X_unk_tabular.append(tmp_feature)

X_unk_tabular = pd.DataFrame(X_unk_tabular)
X_unk_tabular = X_unk_tabular.stack().groupby(level=0).apply(list).tolist()
y_unk_tabular = [1] * len(X_unk_tabular)

X = X_target_tabular + X_unk_tabular
y = y_target_tabular + y_unk_tabular
X, y = shuffle(X, y)

# feature selection
model = XGBClassifier(learning_rate=0.035,
                      colsample_bytree=1,
                      subsample=1,
                      objective='binary:logistic',
                      n_estimators=3000,
                      reg_alpha=0.25,
                      max_depth=7,
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

important_features = []
print("feature ranking")
for i in range(len(importance)):
    print("{}. feature {} ({:.3f}) ".format(i + 1, list(base_feature.keys())[indices[i]], importance[indices[i]]))
    if importance[indices[i]] > 0.002:
        important_features.append(list(base_feature.keys())[indices[i]])
print(len(important_features))

# with open("./important_features.pkl", "wb") as fp:
#     pickle.dump(important_features, fp)

# make featured dataset(train, test)
with open('./important_features.pkl', 'rb') as f:
    important_features = pickle.load(f)

TRAIN_DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2022-09-01~2023-04-14.csv"
TEST_DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2023-04-15~2023-06-08.csv"

train_df = pd.read_csv(TRAIN_DATASET_PATH)
labels = train_df["label"].unique().tolist()

train_target_df = train_df.loc[train_df['label'] != "UNK"].copy()
train_unk_df = train_df.loc[train_df['label'] == "UNK"].copy()

train_target_df.drop(["label", "key"], axis=1, inplace=True)
train_unk_df.drop(["label", "key"], axis=1, inplace=True)

X_train_target = train_target_df.stack().groupby(level=0).apply(list).tolist()
X_train_unk = train_unk_df.stack().groupby(level=0).apply(list).tolist()

X_train_target_tmp = []
MAX_LEN = 20
MIN_LEN = 1
for idx, x in enumerate(X_train_target):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_train_target_tmp.append(x[-MAX_LEN:])
        else:
            X_train_target_tmp.append(x)
X_train_target = X_train_target_tmp

X_train_unk_tmp = []
for idx, x in enumerate(X_train_unk):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_train_unk_tmp.append(x[-MAX_LEN:])
        else:
            X_train_unk_tmp.append(x)
X_train_unk = X_train_unk_tmp

base_i_features = {}
for feature in important_features:
    base_i_features[feature] = 0

X_train_target_tabular = []
for x in X_train_target:
    tmp_feature = base_i_features.copy()
    for item in x:
        if item in important_features:
            tmp_feature[item] += 1
    X_train_target_tabular.append(tmp_feature)

X_train_target_tabular = pd.DataFrame(X_train_target_tabular)
X_train_target_tabular = X_train_target_tabular.stack().groupby(level=0).apply(list).tolist()

X_train_unk_tabular = []
for x in X_train_unk:
    tmp_feature = base_i_features.copy()
    for item in x:
        if item in important_features:
            tmp_feature[item] += 1
    X_train_unk_tabular.append(tmp_feature)

X_train_unk_tabular = pd.DataFrame(X_train_unk_tabular)
X_train_unk_tabular = X_train_unk_tabular.stack().groupby(level=0).apply(list).tolist()

y_train_target = [0] * len(X_train_target_tabular)
y_train_unk = [1] * len(X_train_unk_tabular)

X_train = X_train_target_tabular + X_train_unk_tabular
y_train = y_train_target + y_train_unk
X_train, y_train = shuffle(X_train, y_train)

# test
test_df = pd.read_csv(TEST_DATASET_PATH)

test_target_df = test_df.loc[test_df['label'] != "UNK"].copy()
test_unk_df = test_df.loc[test_df['label'] == "UNK"].copy()

test_target_df.drop(["label", "key"], axis=1, inplace=True)
test_unk_df.drop(["label", "key"], axis=1, inplace=True)

X_test_target = test_target_df.stack().groupby(level=0).apply(list).tolist()
X_test_unk = test_unk_df.stack().groupby(level=0).apply(list).tolist()

X_test_target_tmp = []
MAX_LEN = 20
MIN_LEN = 1
for idx, x in enumerate(X_test_target):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_test_target_tmp.append(x[-MAX_LEN:])
        else:
            X_test_target_tmp.append(x)
X_test_target = X_test_target_tmp

X_test_unk_tmp = []
for idx, x in enumerate(X_test_unk):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_test_unk_tmp.append(x[-MAX_LEN:])
        else:
            X_test_unk_tmp.append(x)
X_test_unk = X_test_unk_tmp

X_test_target_tabular = []
for x in X_test_target:
    tmp_feature = base_i_features.copy()
    for item in x:
        if item in important_features:
            tmp_feature[item] += 1
    X_test_target_tabular.append(tmp_feature)

X_test_target_tabular = pd.DataFrame(X_test_target_tabular)
X_test_target_tabular = X_test_target_tabular.stack().groupby(level=0).apply(list).tolist()

X_test_unk_tabular = []
for x in X_test_unk:
    tmp_feature = base_i_features.copy()
    for item in x:
        if item in important_features:
            tmp_feature[item] += 1
    X_test_unk_tabular.append(tmp_feature)

X_test_unk_tabular = pd.DataFrame(X_test_unk_tabular)
X_test_unk_tabular = X_test_unk_tabular.stack().groupby(level=0).apply(list).tolist()

y_test_target = [0] * len(X_test_target_tabular)
y_test_unk = [1] * len(X_test_unk_tabular)

X_test = X_test_target_tabular + X_test_unk_tabular
y_test = y_test_target + y_test_unk
X_test, y_test = shuffle(X_test, y_test)

# fitting and tuning
hp_grid_rl = [0.25, 0.3, 0.35]
hp_grid_estimators = [500, 800]
hp_grid_m_depth = [4]
hp_grid_spw = [0.8, 5]
hp_grid_gamma = [0.25, 0.3, 0.35]

count = 1
SEARCH_TOTAL = len(hp_grid_rl) * len(hp_grid_estimators) * len(hp_grid_m_depth) * len(hp_grid_spw) * len(hp_grid_gamma)

train_metric_max = 0
test_metric_max = 0

for rl in hp_grid_rl:
    for estimator in hp_grid_estimators:
        for depth in hp_grid_m_depth:
            for spw in hp_grid_spw:
                for gamma in hp_grid_gamma:
                    print(f"{count} / {SEARCH_TOTAL}")
                    model = XGBClassifier(learning_rate=rl,
                                          colsample_bytree=1,
                                          subsample=1,
                                          objective='binary:logistic',
                                          n_estimators=estimator,
                                          reg_alpha=0.25,
                                          max_depth=depth,
                                          scale_pos_weight=spw,
                                          gamma=gamma)
                    pipe = Pipeline([('scaler', RobustScaler()),
                                     ('rf_classifier', model)])
                    pipe.fit(X_train, y_train)
                    # train result
                    pred_x_train = pipe.predict(X_train)
                    train_f1 = f1_score(pred_x_train, y_train, average='binary', zero_division=1)
                    # test result
                    pred_x_test = pipe.predict(X_test)
                    test_f1 = f1_score(pred_x_test, y_test, average='binary', zero_division=1)

                    log = {"grid_search_result": {"hp_grid_gamma": gamma,
                                                  "hp_grid_spw": spw,
                                                  "hp_grid_m_depth": depth,
                                                  "hp_grid_estimators": estimator,
                                                  "hp_grid_rl": rl,
                                                  "F1": train_f1,
                                                  "F1_TEST": test_f1}}
                    print(log)
                    count += 1
                    if train_f1 > train_metric_max and test_f1 > test_metric_max:
                        train_metric_max = train_f1
                        test_metric_max = test_f1

                        filename = "/home/ky/PycharmProjects/pythonProject/b/nbo_xgboost_pipe.sav"
                        joblib.dump(pipe, filename)

                        update_registered_converter(
                            XGBClassifier, 'XGBoostXGBClassifier',
                            calculate_linear_classifier_output_shapes, convert_xgboost,
                            options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

                        model_onnx = convert_sklearn(pipe, 'pipeline_xgb',
                                                     [('input', FloatTensorType([None, len(important_features)]))])

                        # add metadata and save model as onnx
                        meta = model_onnx.metadata_props.add()
                        meta.key = "model_info"
                        cfg = {"input_type": "float", "input_shape": [None, len(important_features)],
                               "labels": {0: "target", 1: "UNK"},
                               "transformer": "nbo.transform_data",
                               "threshold": 0.5,
                               "pos_class": 1}
                        meta.value = str(cfg)

                        now = datetime.now()
                        date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
                        saved_model_path = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/best_result/"
                        log = {"info": {"threshold": 0.5,
                                        "hp_grid_gamma": gamma,
                                        "hp_grid_spw": spw,
                                        "hp_grid_m_depth": depth,
                                        "hp_grid_estimators": estimator,
                                        "hp_grid_rl": rl,
                                        "F1": train_f1,
                                        "F1_TEST": test_f1}}
                        if not os.path.exists(saved_model_path):
                            os.makedirs(saved_model_path)
                        with open(saved_model_path + "nbo_xgboost.onnx", "wb") as f:
                            f.write(model_onnx.SerializeToString())
                        with open(saved_model_path + "/Hparams.json", 'w', encoding="utf-8") as f:
                            json.dump(log, f, ensure_ascii=False, indent=4)



import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import onnxruntime as rt
import numpy as np

NOVELTY_MODEL_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/best_result/nbo_xgboost.onnx"
novelty_session = rt.InferenceSession(NOVELTY_MODEL_PATH)

with open('./important_features.pkl', 'rb') as f:
    nvlt_important_features = pickle.load(f)
base_nvlt_features = {}
for feature in nvlt_important_features:
    base_nvlt_features[feature] = 0

predict_result = []
for idx, data in enumerate(X_test):
    print(f"{idx} / {len(X_test)}")
    pred_novelty = novelty_session.run(None, {"input": np.array([data]).astype(np.float32)})
    pred_novelty = pred_novelty[0]
    if pred_novelty == 0:
        predict_result.append(0)
    else:
        predict_result.append(1)

f1 = f1_score(predict_result, y_test, average=None, zero_division=1)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predict_result)
acc = accuracy_score(y_true=y_test, y_pred=predict_result)
prec = precision_score(y_true=y_test, y_pred=predict_result, average=None)
recl = recall_score(y_true=y_test, y_pred=predict_result, average=None)


# XGB classifier
import pickle
from itertools import chain
import os
from datetime import datetime
import json

import joblib
import pandas as pd
from collections import Counter
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType

# make dataset
DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2022-09-01~2023-04-14.csv"
df = pd.read_csv(DATASET_PATH)
TARGET_LIST = ["EVT0000114", "EVT0000115"]
label_dict = {}

target_df = []
for idx, target in enumerate(TARGET_LIST):
    label_dict[target] = idx
    target_df.append(df.loc[df['label'] == target].copy())

target_df = pd.concat(target_df).reset_index(drop=True)
target_df = target_df.replace({"label": label_dict})

y = target_df[["label"]]
y = y["label"].tolist()
target_df.drop(["label", "key"], axis=1, inplace=True)
X = target_df.stack().groupby(level=0).apply(list).tolist()

X_tmp = []
y_tmp = []
MAX_LEN = 50
MIN_LEN = 1
for idx, x in enumerate(X):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_tmp.append(x[-MAX_LEN:])
        else:
            X_tmp.append(x)
        y_tmp.append(y[idx])

X = X_tmp
y = y_tmp

vocabs = list(set(chain(*X)))
print(f"vocab size: {len(vocabs)} ")

base_feature = {}
for evt in vocabs:
    base_feature[evt] = 0

X_tabular = []
for x in X:
    tmp_feature = base_feature.copy()
    counted_item = Counter(x)
    tmp_feature.update(dict(counted_item))
    X_tabular.append(tmp_feature)

X_tabular = pd.DataFrame(X_tabular)
X_tabular = X_tabular.stack().groupby(level=0).apply(list).tolist()

X_tabular, y = shuffle(X_tabular, y)

# feature selection
model = XGBClassifier(learning_rate=0.035,
                      colsample_bytree=1,
                      subsample=1,
                      objective='binary:logistic',
                      n_estimators=3000,
                      reg_alpha=0.25,
                      max_depth=7,
                      scale_pos_weight=0.7,
                      gamma=0.35)
pipe = Pipeline([('scaler', RobustScaler()),
                 ('rf_classifier', model)])

pipe.fit(X_tabular, y)
pred_x = pipe.predict(X_tabular)

print(f1_score(pred_x, y, average='binary', zero_division=1))
print(roc_auc_score(pred_x, y))
conf_matrix = confusion_matrix(y_true=y, y_pred=pred_x)
print(conf_matrix.ravel())
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

important_features = []
print("feature ranking")
for i in range(len(importance)):
    # print("{}. feature {} ({:.3f}) ".format(i + 1, list(base_feature.keys())[indices[i]], importance[indices[i]]))
    if importance[indices[i]] > 0.002:
        important_features.append(list(base_feature.keys())[indices[i]])
print(len(important_features))

with open("./important_features_target_classifier.pkl", "wb") as fp:
    pickle.dump(important_features, fp)

# with open('./important_features_target_classifier.pkl', 'rb') as f:
#     important_features = pickle.load(f)

# make featured dataset
TRAIN_DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2022-09-01~2023-04-14.csv"
TEST_DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2023-04-15~2023-06-08.csv"

X_train_df = pd.read_csv(TRAIN_DATASET_PATH)

X_train_df_tmp = []
for idx, target in enumerate(TARGET_LIST):
    X_train_df_tmp.append(X_train_df.loc[X_train_df['label'] == target].copy())

X_train_df = pd.concat(X_train_df_tmp).reset_index(drop=True)
X_train_df = X_train_df.replace({"label": label_dict})

y_train = X_train_df[["label"]]
y_train = y_train["label"].tolist()
X_train_df.drop(["label", "key"], axis=1, inplace=True)
X_train = X_train_df.stack().groupby(level=0).apply(list).tolist()

X_tmp = []
y_tmp = []
for idx, x in enumerate(X_train):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_tmp.append(x[-MAX_LEN:])
        else:
            X_tmp.append(x)
        y_tmp.append(y_train[idx])

X_train = X_tmp
y_train = y_tmp

base_i_features = {}
for feature in important_features:
    base_i_features[feature] = 0

X_train_tabular = []
for x in X_train:
    tmp_feature = base_i_features.copy()
    for item in x:
        if item in important_features:
            tmp_feature[item] += 1
    X_train_tabular.append(tmp_feature)

X_train_tabular = pd.DataFrame(X_train_tabular)
X_train = X_train_tabular.stack().groupby(level=0).apply(list).tolist()

X_train, y_train = shuffle(X_train, y_train)

# test set
X_test_df = pd.read_csv(TEST_DATASET_PATH)

X_test_df_tmp = []
for idx, target in enumerate(TARGET_LIST):
    X_test_df_tmp.append(X_test_df.loc[X_test_df['label'] == target].copy())

X_test_df_tmp = pd.concat(X_test_df_tmp).reset_index(drop=True)
X_test_df = X_test_df_tmp.replace({"label": label_dict})

y_test = X_test_df[["label"]]
y_test = y_test["label"].tolist()
X_test_df.drop(["label", "key"], axis=1, inplace=True)
X_test = X_test_df.stack().groupby(level=0).apply(list).tolist()

X_tmp = []
y_tmp = []
for idx, x in enumerate(X_test):
    x_len = len(x)
    if x_len >= MIN_LEN:
        if x_len >= MAX_LEN:
            X_tmp.append(x[-MAX_LEN:])
        else:
            X_tmp.append(x)
        y_tmp.append(y_test[idx])

X_test = X_tmp
y_test = y_tmp

base_i_features = {}
for feature in important_features:
    base_i_features[feature] = 0

X_test_tabular = []
for x in X_test:
    tmp_feature = base_i_features.copy()
    for item in x:
        if item in important_features:
            tmp_feature[item] += 1
    X_test_tabular.append(tmp_feature)

X_test_tabular = pd.DataFrame(X_test_tabular)
X_test = X_test_tabular.stack().groupby(level=0).apply(list).tolist()

X_test, y_test = shuffle(X_test, y_test)


# fitting and tuning
hp_grid_rl = [0.35, 0.25]
hp_grid_estimators = [800]
hp_grid_m_depth = [7]
hp_grid_spw = [1.2]
hp_grid_gamma = [0.3]

count = 1
SEARCH_TOTAL = len(hp_grid_rl) * len(hp_grid_estimators) * len(hp_grid_m_depth) * len(hp_grid_spw) * len(hp_grid_gamma)

train_metric_max = 0
test_metric_max = 0

for rl in hp_grid_rl:
    for estimator in hp_grid_estimators:
        for depth in hp_grid_m_depth:
            for spw in hp_grid_spw:
                for gamma in hp_grid_gamma:
                    print(f"{count} / {SEARCH_TOTAL}")
                    model = XGBClassifier(learning_rate=rl,
                                          colsample_bytree=1,
                                          subsample=1,
                                          objective='binary:logistic',
                                          n_estimators=estimator,
                                          reg_alpha=0.25,
                                          max_depth=depth,
                                          scale_pos_weight=spw,
                                          gamma=gamma)
                    pipe = Pipeline([('scaler', RobustScaler()),
                                     ('rf_classifier', model)])
                    pipe.fit(X_train, y_train)
                    # train result
                    pred_x_train = pipe.predict(X_train)
                    train_f1 = f1_score(pred_x_train, y_train, average='binary', zero_division=1)
                    # test result
                    pred_x_test = pipe.predict(X_test)
                    test_f1 = f1_score(pred_x_test, y_test, average='binary', zero_division=1)

                    log = {"grid_search_result": {"hp_grid_gamma": gamma,
                                                  "hp_grid_spw": spw,
                                                  "hp_grid_m_depth": depth,
                                                  "hp_grid_estimators": estimator,
                                                  "hp_grid_rl": rl,
                                                  "F1": train_f1,
                                                  "F1_TEST": test_f1}}
                    print(log)
                    count += 1
                    if test_f1 > test_metric_max:
                        train_metric_max = train_f1
                        test_metric_max = test_f1

                        filename = "/home/ky/PycharmProjects/pythonProject/b/nbo_cls_xgboost_pipe.sav"
                        joblib.dump(pipe, filename)

                        update_registered_converter(
                            XGBClassifier, 'XGBoostXGBClassifier',
                            calculate_linear_classifier_output_shapes, convert_xgboost,
                            options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

                        model_onnx = convert_sklearn(pipe, 'pipeline_xgb',
                                                     [('input', FloatTensorType([None, len(important_features)]))])

                        # add metadata and save model as onnx
                        meta = model_onnx.metadata_props.add()
                        meta.key = "model_info"
                        cfg = {"input_type": "float", "input_shape": [None, len(important_features)],
                               "labels": {0: "EVT0000114", 1: "EVT0000115"},
                               "transformer": "nbo.transform_data",
                               "threshold": 0.5}
                        meta.value = str(cfg)

                        now = datetime.now()
                        date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
                        saved_model_path = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_cls_test/best_result/"
                        log = {"info": {"threshold": 0.5,
                                        "hp_grid_gamma": gamma,
                                        "hp_grid_spw": spw,
                                        "hp_grid_m_depth": depth,
                                        "hp_grid_estimators": estimator,
                                        "hp_grid_rl": rl,
                                        "F1": train_f1,
                                        "F1_TEST": test_f1}}
                        if not os.path.exists(saved_model_path):
                            os.makedirs(saved_model_path)
                        with open(saved_model_path + "nbo_cls_xgboost.onnx", "wb") as f:
                            f.write(model_onnx.SerializeToString())
                        with open(saved_model_path + "/cls_Hparams.json", 'w', encoding="utf-8") as f:
                            json.dump(log, f, ensure_ascii=False, indent=4)



import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import onnxruntime as rt
import numpy as np

CLS_MODEL_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_cls_test/best_result/nbo_cls_xgboost.onnx"
cls_session = rt.InferenceSession(CLS_MODEL_PATH)


predict_result = []

for idx, data in enumerate(X_test):
    print(f"{idx} / {len(X_test)}")
    pred_novelty = cls_session.run(None, {"input": np.array([data]).astype(np.float32)})
    pred_novelty = pred_novelty[0]
    if pred_novelty == 0:
        predict_result.append(0)
    else:
        predict_result.append(1)

f1 = f1_score(predict_result, y_test, average=None, zero_division=1)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predict_result)
acc = accuracy_score(y_true=y_test, y_pred=predict_result)
prec = precision_score(y_true=y_test, y_pred=predict_result, average=None)
recl = recall_score(y_true=y_test, y_pred=predict_result, average=None)


# combined model
# load dataset (test, actual)
# load test dataset
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import onnxruntime as rt
import numpy as np

TEST_DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/2023-04-15~2023-06-08.csv"
test_df = pd.read_csv(TEST_DATASET_PATH)
labels = {"UNK": 0, "EVT0000114": 1, "EVT0000115": 2}
test_df = test_df.replace({"label": labels})
y_test = test_df["label"].tolist()
test_df.drop(["label", "key"], axis=1, inplace=True)
X_test = test_df.stack().groupby(level=0).apply(list).tolist()

# load req dataset
REQ_DATASET_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/actual.txt"
req_ds = open(REQ_DATASET_PATH, 'r', encoding='utf-16')
Lines = req_ds.readlines()
req_data = []
for idx, line in enumerate(Lines):
    line = line.rstrip("\n")
    if idx == 0:
        continue
    line = eval(line)
    if isinstance(line, list):
        req_data.append(line)
    elif isinstance(line, dict):
        req_data.append(line["INPUT"])

# load model - novelty
NOVELTY_MODEL_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_test/best_result/nbo_xgboost.onnx"
CLS_MODEL_PATH = "/home/ky/PycharmProjects/pythonProject/dataset/nbo_cls_test/best_result/nbo_cls_xgboost.onnx"
novelty_session = rt.InferenceSession(NOVELTY_MODEL_PATH)
cls_session = rt.InferenceSession(CLS_MODEL_PATH)

with open('./important_features.pkl', 'rb') as f:
    nvlt_important_features = pickle.load(f)
base_nvlt_features = {}
for feature in nvlt_important_features:
    base_nvlt_features[feature] = 0

with open('./important_features_target_classifier.pkl', 'rb') as f:
    cls_important_features = pickle.load(f)
base_cls_features = {}
for feature in cls_important_features:
    base_cls_features[feature] = 0
predict_result = []

for idx, data in enumerate(X_test):
    print(f"{idx} / {len(X_test)}")
    # convert data to tabular
    n_data = data
    if len(n_data) > 20:
        n_data = n_data[20:]

    tmp_feature = base_nvlt_features.copy()
    for item in n_data:
        if item in base_nvlt_features:
            tmp_feature[item] += 1
    n_data = list(tmp_feature.values())
    # predict
    pred_novelty = novelty_session.run(None, {"input": np.array([n_data]).astype(np.float32)})
    pred_novelty = pred_novelty[0]
    if pred_novelty == 0:
        c_data = data
        if len(c_data) > 50:
            c_data = c_data[50:]
        tmp_feature = base_cls_features.copy()
        for item in c_data:
            if item in base_cls_features:
                tmp_feature[item] += 1
        c_data = list(tmp_feature.values())

        pred_cls = cls_session.run(None, {"input": np.array([c_data]).astype(np.float32)})
        pred_cls = pred_cls[0]
        if pred_cls == 0:
            predict_result.append(1)
        else:
            predict_result.append(2)
    else:
        predict_result.append(0)

f1 = f1_score(predict_result, y_test, average=None, zero_division=1)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predict_result)
acc = accuracy_score(y_true=y_test, y_pred=predict_result)
prec = precision_score(y_true=y_test, y_pred=predict_result, average=None)
recl = recall_score(y_true=y_test, y_pred=predict_result, average=None)


# req data result
req_predict_result = []
for idx, data in enumerate(req_data):
    print(f"{idx} / {len(req_data)}")
    # convert data to tabular
    n_data = data
    if len(n_data) > 20:
        n_data = n_data[20:]

    tmp_feature = base_nvlt_features.copy()
    for item in n_data:
        if item in base_nvlt_features:
            tmp_feature[item] += 1
    n_data = list(tmp_feature.values())
    # predict
    pred_novelty = novelty_session.run(None, {"input": np.array([n_data]).astype(np.float32)})
    pred_novelty = pred_novelty[0]
    if pred_novelty == 0:
        c_data = data
        if len(c_data) > 50:
            c_data = c_data[50:]
        tmp_feature = base_cls_features.copy()
        for item in c_data:
            if item in base_cls_features:
                tmp_feature[item] += 1
        c_data = list(tmp_feature.values())

        pred_cls = cls_session.run(None, {"input": np.array([c_data]).astype(np.float32)})
        pred_cls = pred_cls[0]
        if pred_cls == 0:
            req_predict_result.append(1)
        else:
            req_predict_result.append(2)
    else:
        req_predict_result.append(0)

unique, counts = np.unique(req_predict_result, return_counts=True)
req_res = dict(zip(unique, counts))
req_res_t = sum(req_res.values())
res_ratio = {key: (value / req_res_t) for key, value in req_res.items()}

# remaining task : reporting each approach(+default model)
