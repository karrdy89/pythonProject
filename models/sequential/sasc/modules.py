import tensorflow as tf

def multihead_attention(queries, keys, attention_dim: int, num_heads: int, dropout_rate: float):
    # Linear projections
    Q = tf.keras.layers.Dense(attention_dim, activation=None)(queries) # (N, T_q, C)
    K = tf.keras.layers.Dense(attention_dim, activation=None)(keys) # (N, T_k, C)
    V = tf.keras.layers.Dense(attention_dim, activation=None)(keys) # (N, T_k, C)
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
        return dict(list(base_config.items())+list(config.items()))
