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
import numpy as np
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
X = df.stack().groupby(level=0).apply(list).tolist()

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
embedding_dim = 64
l2_reg = 1e-6
dropout_rate = 0.3
num_blocks = 2
attention_dim = 64
attention_num_heads = 8
conv_dims = [64, 64]
epsilon = 1e-08
learning_rate = 0.0013

inputs = layers.Input(shape=(None,), name='seq', dtype=object)
encoding_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=events, mask_token=mask)
encoded_inputs = encoding_layer(inputs)
positional_embedding_layer = tf.keras.layers.Embedding(max_len, embedding_dim,
                                                       name='positional_embeddings', mask_zero=False,
                                                       embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
item_embedding_layer = tf.keras.layers.Embedding(vocab_size+1, embedding_dim,
                                                 name='item_embeddings', mask_zero=True,
                                                 embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
seq_embeddings, positional_embeddings = embedding(encoded_inputs, item_embedding_layer, positional_embedding_layer)
seq_embeddings += positional_embeddings
mask = tf.expand_dims(tf.cast(tf.not_equal(encoded_inputs, 0), tf.float32), -1)
seq_embeddings = tf.keras.layers.Dropout(dropout_rate)(seq_embeddings)
seq_embeddings *= mask
seq_attention = seq_embeddings
encoder = Encoder(num_blocks, max_len, embedding_dim, attention_dim, attention_num_heads, conv_dims, dropout_rate,)
seq_attention = encoder(seq_attention, True, mask)
layer_normalization = LayerNormalization(max_len, embedding_dim, epsilon)
x = layer_normalization(seq_attention)  # (b, s, d)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(num_labels, activation="softmax")(x)
model = tf.keras.Model(inputs=[inputs], outputs=outputs)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                     amsgrad=False)


model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=10, batch_size=64)





#
# n_hiddens = 64
# embed_dim = 20  # Embedding size for each token
# # inputs = layers.Input(shape=(max_len,), name='seq')
# inputs = layers.Input(shape=(None,), name='seq', dtype=object)
# encoding_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=events, mask_token=mask)
# encoded_inputs = encoding_layer(inputs)
# x1 = tf.keras.layers.Embedding(vocab_size+1, embed_dim, input_length=max_len, mask_zero=True)(encoded_inputs)
# x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_hiddens))(x1)
# x1 = tf.keras.layers.Dense(num_labels, activation='softmax', name='output')(x1)
# lstm_model = tf.keras.models.Model(inputs=[inputs], outputs=x1)
# lstm_model.summary()
# lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
# lstm_history = lstm_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=5, batch_size=32)


#
# from models.sequential.sasc.modules_2 import *
# from tensorflow.python.keras.regularizers import l2
#
# num_layers = 2  # Number of encoding layer
# dropout = 0.3  # dropout rate of embedding layer
# embed_dim = 20  # Embedding size for each token
# num_heads = 5  # Number of attention heads
# ff_dim = 64  # Hidden layer size in feed forward network inside transformer
# learning_rate = 0.0013  # Learning rate of optimizer
#
#
# inputs = layers.Input(shape=(None,), name='seq', dtype=object)
# padding_layer = PaddingLayer(maxlen=max_len, name='padding')
# padding_input = padding_layer(inputs)
# encoding_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=events)
# encoded_inputs = encoding_layer(padding_input)
# embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len, embeddings_initializer='random_uniform', embeddings_regularizer=l2(1e-6))(encoded_inputs)
# embeddings *= tf.math.sqrt(tf.cast(embed_dim, tf.float32))
# embeddings = PositionalEncoding(vocab_size, embed_dim)(embeddings)
# embeddings = tf.keras.layers.Dropout(rate=dropout)(embeddings)
# enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1,1,None), name='enc_padding_mask')(encoded_inputs)
# transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
# x = transformer_block(embeddings, enc_padding_mask)
# for i in range(num_layers-1):
#     x = transformer_block(x, enc_padding_mask)
# x = layers.GlobalAveragePooling1D()(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Dense(64, activation="relu")(x)
# x = layers.Dense(16, activation="relu")(x)
# x = layers.Dropout(0.1)(x)
# outputs = layers.Dense(num_labels, activation="softmax")(x)
# model = tf.keras.Model(inputs=[inputs], outputs=outputs)
# model.summary()
#
# optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#
# def loss_function(y_true, y_pred):
#   loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
#   mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
#   loss = tf.multiply(loss, mask)
#   return tf.reduce_mean(loss)
#
#
# model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
# history = model.fit(X_train, y_train, validation_data=(X_test,y_test),epochs=3, batch_size=64)
