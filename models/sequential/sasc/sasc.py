# import pandas as pd
# import numpy as np
# from tensorflow.keras.layers.experimental import preprocessing
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from models.sequential.sasc.modules import *
#
# df = pd.read_csv("dataset/CJ_train.csv")
# labels = df['Target'].unique()
# labels = labels.tolist()
# samples = {}
# for i in labels:
#     samples[i] = df["Target"].value_counts()[i]
# min_sample = min(samples, key=samples.get)
#
# sep_frames = []
# for label in labels:
#     if label is not min_sample:
#         downed = df[df['Target'] == label]
#         downed = downed.sample(n=samples[min_sample].item(), random_state=0)
#         sep_frames.append(downed)
#
# df = pd.concat(sep_frames, axis=0)
# df.fillna('', inplace=True)
#
# y = df.pop("Target")
# df = df.iloc[:, ::-1]
# X = df.stack().groupby(level=0).apply(list).tolist()
#
# oneCol = []
# for k in df:
#     oneCol.append(df[k])
# combined = pd.concat(oneCol, ignore_index=True)
#
# events = combined.unique().tolist()
# mask = events.pop(events.index(''))
# le = preprocessing.StringLookup()
# le.adapt(events)
#
# label_vocab = preprocessing.IntegerLookup(dtype='int64')
# label_vocab.adapt(labels)
#
# X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=0.5, shuffle=False)
#
# X_train = np.array(X_train, dtype='object')
# X_valid = np.array(X_valid, dtype='object')
# X_test = np.array(X_test, dtype='object')
# y_train = np.array(y_train).astype(int)
# y_train = label_vocab(y_train)
# y_valid = np.array(y_valid).astype(int)
# y_valid = label_vocab(y_valid)
# y_test = np.array(y_test).astype(int)
# y_test = label_vocab(y_test)
#
# vocab_size = len(le.get_vocabulary())  # Size of sequence vocab
# max_len = len(df.columns)  # Sequence length
# num_labels = len(label_vocab.get_vocabulary())
#

def get_model(vocab_size:int, max_len:int, num_labels:int, embedding_dim: int = 64, dropout_rate: float = 0.3,
              num_blocks: int = 2, attention_num_heads: int = 8, l2_reg: float = 1e-6, epsilon: float = 1e-8,
              learning_rate: float = 0.0013, mask_token: str = ''):
    attention_dim = embedding_dim
    conv_dims = [embedding_dim, embedding_dim]
    inputs = layers.Input(shape=(None,), name='seq', dtype=object)
    encoding_layer = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=events, mask_token=mask_token)
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



# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
