# plot data distribution
# preprocess
# make dataset
# build model
# train
# check result
# add each stop to pipeline

# import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt


# write model
# if distribution pipe -> store(best param) -> split train function

import pandas as pd
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from models.sequential.sasc.modules import *

df = pd.read_csv("dataset/CJ_train.csv")

df.head(5)
labels = df['Target'].unique()
labels = labels.tolist()
label_vocab = preprocessing.IntegerLookup(dtype='int64')
label_vocab.adapt(labels)
samples = {}
for i in labels:
    samples[i] = df["Target"].value_counts()[i]
min_sample = min(samples, key=samples.get)

sep_frames = []
for label in labels:
    if label is not min_sample:
        downed = df[df['Target'] == label]
        downed = downed.sample(n=samples[min_sample].item(), random_state=0)
        sep_frames.append(downed)

df = pd.concat(sep_frames, axis=0)
df.fillna('', inplace=True)

y = df.pop("Target")
df = df.iloc[:, ::-1]
# df_list = df.values.tolist()
X = df.stack().groupby(level=0).apply(list).tolist()
# X = []
# for j in df_list:
#     X.append(j)

oneCol = []
for k in df:
    oneCol.append(df[k])
combined = pd.concat(oneCol, ignore_index=True)

events = combined.unique().tolist()
mask = events.pop(events.index(''))
le = preprocessing.StringLookup()
le.adapt(events)

X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.5, shuffle=False)

X_train = np.array(X_train, dtype='object')
X_valid = np.array(X_valid, dtype='object')
X_test = np.array(X_test, dtype='object')
y_train = np.array(y_train).astype(int)
y_train = label_vocab(y_train)
y_valid = np.array(y_valid).astype(int)
y_valid = label_vocab(y_valid)
y_test = np.array(y_test).astype(int)
y_test = label_vocab(y_test)

vocab_size = len(le.get_vocabulary())  # Size of sequence vocab
max_len = len(df.columns)  # Sequence length
num_labels = len(label_vocab.get_vocabulary())
embed_dim = 10
l2_reg = 1e-6
dropout_rate = 0.4
num_blocks = 2
attention_dim = 64
attention_num_heads = 5

inputs = layers.Input(shape=(None,), name='seq', dtype=object)
encoding_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=events, mask_token=mask)
encoded_inputs = encoding_layer(inputs)
positional_embedding_layer = tf.keras.layers.Embedding(max_len, embed_dim,
                                                       name='positional_embeddings', mask_zero=False,
                                                       embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
item_embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim,
                                                 name='item_embeddings', mask_zero=True,
                                                 embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
seq_embeddings, positional_embeddings = embedding(encoded_inputs, item_embedding_layer, positional_embedding_layer)
seq_embeddings += positional_embeddings
mask = tf.expand_dims(tf.cast(tf.not_equal(encoded_inputs, 0), tf.float32), -1)
seq_embeddings = tf.keras.layers.Dropout(dropout_rate)(seq_embeddings)
seq_embeddings *= mask
seq_attention = seq_embeddings
for i in range(num_blocks):
    seq_attention = multihead_attention(queries=layer_normalization(seq_attention),
                                        keys=seq_attention,
                                        attention_dim=attention_dim,
                                        num_heads=attention_num_heads,
                                        dropout_rate=dropout_rate)
    seq_attention = point_wise_feed_forward(seq_attention, dropout_rate=dropout_rate, conv_dims=conv_dims)
    seq_attention *= mask


model = tf.keras.Model(inputs=[inputs], outputs=encoded_inputs)
model.predict([X_train[0]])
