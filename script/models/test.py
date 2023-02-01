# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1. read data from db
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import itertools

import oracledb
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# db connection info
# USER = "S_AIB_U"
# PASSWORD = "Csxdsxzes4#"
# IP = "10.10.10.42"
# PORT = 1525
# SID = "FDSD"

USER = "SYSTEM"
PASSWORD = "oracle1234"
IP = "192.168.113.1"
PORT = 3000
SID = "KY"

dsn = oracledb.makedsn(host=IP, port=PORT, sid=SID)
session_pool = oracledb.SessionPool(user=USER, password=PASSWORD, dsn=dsn,
                                    min=2, max=10,
                                    increment=1, encoding="UTF-8")

# check db connection
try:
    test_connection = session_pool.acquire()
except Exception as exc:
    print("connection fail")
    raise exc
else:
    print("connection success")
    session_pool.release(test_connection)

# read data from db
START_DATE = "20220601"
END_DATE = "20230130"
# query = "SELECT /*+ INDEX_DESC(A IXTBCHN3001H03) */ \
#                 A.CUST_NO \
#                 , A.ORGN_DTM \
#                 , A.CHNL_ID \
#                 , A.EVNT_ID \
#             FROM \
#                 S_AIB.TBCHN3001H A \
#             WHERE 1=1 \
#             AND CUST_NO IS NOT NULL \
#             AND ORGN_DTM BETWEEN " + START_DATE + " || '000000' AND " + END_DATE + " || '999999'"

query = "SELECT CUST_NO, EVNT_ID \
        FROM TEST \
        WHERE 1=1 AND ORGN_DTM BETWEEN TO_DATE(" + START_DATE + ", 'YYYYMMDD') AND TO_DATE(" + END_DATE + ", 'YYYYMMDD') ORDER BY CUST_NO ASC, ORGN_DTM ASC"

with session_pool.acquire() as conn:
    cursor = conn.cursor()
    cursor.execute(query)
    df = pd.DataFrame(cursor.fetchall())
    df.columns = [x[0] for x in cursor.description]
    cursor.close()

assert len(df) != 0, "No value retrieved"
print(df.head(10))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2. transform data
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
LABELS = ["EVT0000001", "EVT0000002", "EVT0000003", "UNK"]
NUM_SAMPLE = 100
MAX_LEN = 50

# transform data
cust_nos = df["CUST_NO"].unique().tolist()
samples_x = []
samples_y = []
labels_ratio = {}
t_max_len = 0
is_finished = False
total_label_num = len(LABELS)
data_per_label = int(NUM_SAMPLE / total_label_num)
diff = NUM_SAMPLE - data_per_label * total_label_num
for label in LABELS:
    if label == "UNK":
        labels_ratio[label] = data_per_label + diff
    else:
        labels_ratio[label] = data_per_label

print(labels_ratio)

for idx, cust_no in enumerate(cust_nos):
    if is_finished:
        break
    print(f"{idx} / {len(cust_nos)}", end='\r')
    df_t = df[df["CUST_NO"] == cust_no].reset_index(drop=True)
    list_t = df_t["EVNT_ID"].tolist()
    b_idx = 0
    if len(list_t) > 2:
        for idx_t, event in enumerate(list_t):
            if len(samples_x) >= NUM_SAMPLE:
                is_finished = True
                break
            if event in LABELS:
                if event in labels_ratio:
                    if idx_t - b_idx > 1:
                        sample = list_t[b_idx:idx_t]
                        c_max_len = len(sample)
                        if c_max_len > MAX_LEN:
                            c_max_len = MAX_LEN
                            sample = list_t[idx_t - MAX_LEN:idx_t]
                        if c_max_len > t_max_len:
                            t_max_len = c_max_len
                        samples_x.append(sample)
                        samples_y.append(event)
                        labels_ratio[event] = labels_ratio[event] - 1
                        if labels_ratio[event] <= 0:
                            del labels_ratio[event]
                        b_idx = idx_t
            else:
                if "UNK" in labels_ratio:
                    if idx_t != 0:
                        sample = list_t[0:idx_t]
                        samples_x.append(sample)
                        samples_y.append("UNK")
                        labels_ratio["UNK"] = labels_ratio["UNK"]-1
                        if labels_ratio["UNK"] <= 0:
                            del labels_ratio["UNK"]

assert len(samples_x) >= 10, "minimum sample size is 10"


def idx_mapping(val):
    return LABELS.index(val)


samples_y = list(map(idx_mapping, samples_y))

# padding
for idx, sample in enumerate(samples_x):
    samples_x[idx] = sample + [""] * (t_max_len - len(sample))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 3. define model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def multihead_attention(queries, keys, attention_dim: int, num_heads: int, dropout_rate: float):
    Q = tf.keras.layers.Dense(attention_dim, activation=None)(queries)
    K = tf.keras.layers.Dense(attention_dim, activation=None)(keys)
    V = tf.keras.layers.Dense(attention_dim, activation=None)(keys)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
    key_masks = tf.tile(key_masks, [num_heads, 1])
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

    diag_vals = tf.ones_like(outputs[0, :, :])
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
    masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
    paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    outputs = tf.nn.softmax(outputs)
    query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
    query_masks = tf.tile(query_masks, [num_heads, 1])
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
    outputs *= query_masks

    outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
    outputs = tf.matmul(outputs, V_)

    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
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


class LabelLayer(tf.keras.layers.Layer):
    def __init__(self, labels, topn, **kwargs):
        self.labels = labels
        self.topn = topn
        super(LabelLayer, self).__init__(**kwargs)

    def call(self, x):
        batchsize = tf.shape(x)[0]
        tf_labels = tf.constant([self.labels], dtype='string')
        tf_labels = tf.tile(tf_labels, [batchsize, 1])
        top_k = tf.nn.top_k(x, k=self.topn, sorted=True, name='top_k').indices
        top_label = tf.gather(tf_labels, top_k, batch_dims=1)
        conf = tf.sort(x, axis=-1, direction='DESCENDING')
        top_label = tf.squeeze(top_label)
        conf = tf.squeeze(conf)
        return top_label, conf

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        top_shape = (batch_size, len(self.labels))
        return top_shape

    def get_config(self):
        config = {'labels': self.labels}
        base_config = super(LabelLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_model(vocab_size: int, vocabulary, max_len: int, num_labels: int, embedding_dim: int = 64,
              dropout_rate: float = 0.3, num_blocks: int = 2, attention_num_heads: int = 8, l2_reg: float = 1e-6,
              epsilon: float = 1e-8, learning_rate: float = 0.0013, mask_token: str = ''):
    attention_dim = embedding_dim
    conv_dims = [embedding_dim, embedding_dim]

    input_meta = {"max_len": max_len, "transformer": "nbo.transform_data"}
    input_meta = encode_tf_input_meta(input_meta)
    inputs = layers.Input(shape=(None,), name='input_meta' + input_meta, dtype=object)
    encoding_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocabulary,
                                                                             mask_token=mask_token)
    encoded_inputs = encoding_layer(inputs)
    positional_embedding_layer = tf.keras.layers.Embedding(max_len, embedding_dim,
                                                           name='positional_embeddings', mask_zero=False,
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
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_labels, activation="softmax")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                         amsgrad=False)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    return model


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 4. train and save
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
SPLIT_RATIO = "80:10:10"
BATCH_SIZE = 32
EPOCH = 3


def encode_tf_input_meta(meta: dict) -> str:
    meta = str(meta)
    meta = meta.replace(" ", '')
    meta = meta.replace("{'", "_spds__spk_").replace('}', "_spde_") \
        .replace("':", "_spv_").replace(",'", "_spk_").replace("'", "_spq_").replace(".", "_spd_")
    return meta


def split_ratio(data_split: str):
    ratio = data_split.split(":")
    train = float(ratio[0]) * 0.01
    val = float(ratio[1]) * 0.01
    test = float(ratio[2]) * 0.01
    return train, val, test


max_len = MAX_LEN + 1
vocabs = list(set(list(itertools.chain.from_iterable(samples_x))))

mask_token = vocabs.pop(vocabs.index(""))
le = preprocessing.StringLookup()
le.adapt(vocabs)

vocab_size = len(le.get_vocabulary()) + 1
num_labels = len(LABELS)
model = get_model(vocab_size=vocab_size, vocabulary=vocabs, max_len=max_len, num_labels=num_labels,
                  mask_token=mask_token)

train_ratio, validation_ratio, test_ratio = split_ratio(SPLIT_RATIO)
X_train, X_valid_test, y_train, y_valid_test = train_test_split(samples_x, samples_y,
                                                                test_size=validation_ratio + test_ratio,
                                                                shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test,
                                                    test_size=(1 - test_ratio / (
                                                            validation_ratio + test_ratio)),
                                                    shuffle=True)
X_train = np.array(X_train, dtype='object')
y_train = np.array(y_train).astype(np.int)

X_valid = np.array(X_valid, dtype='object')
y_valid = np.array(y_valid).astype(np.int)

X_test = np.array(X_test, dtype='object')
y_test = np.array(y_test).astype(np.int)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH,
          batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=1)

model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
labeled_output = LabelLayer(LABELS, num_labels, name='result')(model.outputs)
model = tf.keras.Model(model.input, labeled_output)
model.save("/tmp/")

# model.predict(tf.constant([X_train[1]]))
