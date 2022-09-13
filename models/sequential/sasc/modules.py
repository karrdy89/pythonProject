import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D

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

def layer_normalization(input_seq, epsilon = 1e-8):

    inputs_shape = input_seq.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(input_seq, [-1], keep_dims=True)
    beta = tf.zeros(params_shape)
    gamma = tf.ones(params_shape)
    normalized = (input_seq - mean) / ((variance + epsilon) ** .5)
    output = gamma * normalized + beta

    return output


class PointWiseFeedForward(tf.keras.layers.Layer):
    """
    Convolution layers with residual connection
    """

    def __init__(self, conv_dims, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv_dims = conv_dims
        self.dropout_rate = dropout_rate
        self.conv_layer1 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[0], kernel_size=1, activation="relu", use_bias=True
        )
        self.conv_layer2 = tf.keras.layers.Conv1D(
            filters=self.conv_dims[1], kernel_size=1, activation=None, use_bias=True
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x):

        output = self.conv_layer1(x)
        output = self.dropout_layer(output)

        output = self.conv_layer2(output)
        output = self.dropout_layer(output)

        # Residual connection
        output += x

        return output


def embedding(input_seq, item_embedding_layer, positional_embedding_layer):
    seq_embeddings = item_embedding_layer(input_seq)
    positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
    positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
    positional_embeddings = positional_embedding_layer(positional_seq)
    return seq_embeddings, positional_embeddings
