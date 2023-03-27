from models.sequential.sasc.modules import *
from tensorflow.keras import layers
from utils.common import encode_tf_input_meta


def get_model(vocab_size: int, vocabulary, max_len: int, num_labels: int, embedding_dim: int = 32,
              dropout_rate: float = 0.2, num_blocks: int = 1, attention_num_heads: int = 4, l2_reg: float = 1e-6,
              epsilon: float = 1e-8, learning_rate: float = 0.012, mask_token: str = ''):
    attention_dim = embedding_dim
    conv_dims = [embedding_dim, embedding_dim]

    input_meta = {"max_len": max_len, "transformer": "nbo.transform_data"}
    input_meta = encode_tf_input_meta(input_meta)
    inputs = layers.Input(shape=(None,), name='input_meta'+input_meta, dtype=object)
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
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_labels, activation="softmax")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                         amsgrad=False)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    return model
