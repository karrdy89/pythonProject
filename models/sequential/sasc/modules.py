import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    - Q (query), K (key) and V (value) are split into multiple heads (num_heads)
    - each tuple (q, k, v) are fed to scaled_dot_product_attention
    - all attention outputs are concatenated
    """

    def __init__(self, attention_dim, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        assert attention_dim % self.num_heads == 0
        self.dropout_rate = dropout_rate

        self.depth = attention_dim // self.num_heads

        self.Q = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.K = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.V = tf.keras.layers.Dense(self.attention_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, queries, keys):
        # Linear projections
        Q = self.Q(queries)  # (N, T_q, C)
        K = self.K(keys)  # (N, T_k, C)
        V = self.V(keys)  # (N, T_k, C)

        # --- MULTI HEAD ---
        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # --- SCALED DOT PRODUCT ---
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # print(outputs.shape)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)

        key_masks = tf.tile(key_masks, [self.num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(
            tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Future blinding (Causality)
        diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(
            diag_vals
        ).to_dense()  # (T_q, T_k)
        masks = tf.tile(
            tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(masks) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [self.num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(
            tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]
        )  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # --- MULTI HEAD ---
        # concat heads
        outputs = tf.concat(
            tf.split(outputs, self.num_heads, axis=0), axis=2
        )  # (N, T_q, C)

        # Residual connection
        outputs += queries

        return outputs


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


class EncoderLayer(tf.keras.layers.Layer):
    """
    Transformer based encoder layer
    """

    def __init__(
            self,
            seq_max_len,
            embedding_dim,
            attention_dim,
            num_heads,
            conv_dims,
            dropout_rate,
    ):
        super(EncoderLayer, self).__init__()

        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim

        self.mha = MultiHeadAttention(attention_dim, num_heads, dropout_rate)
        self.ffn = PointWiseFeedForward(conv_dims, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_normalization = CustomLayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def call_(self, x, training, mask):
        attn_output = self.mha(queries=self.layer_normalization(x), keys=x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # feed forward network
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        # masking
        out2 *= mask
        return out2

    def call(self, x, training, mask):
        x_norm = self.layer_normalization(x)
        attn_output = self.mha(queries=x_norm, keys=x)
        attn_output = self.ffn(attn_output)
        out = attn_output * mask
        return out

    def get_config(self):
        cfg = super().get_config()
        return cfg


class Encoder(tf.keras.layers.Layer):
    """
    Invokes Transformer based encoder with user defined number of layers
    """

    def __init__(
            self,
            num_layers,
            seq_max_len,
            embedding_dim,
            attention_dim,
            num_heads,
            conv_dims,
            dropout_rate,
    ):
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [
            EncoderLayer(
                seq_max_len,
                embedding_dim,
                attention_dim,
                num_heads,
                conv_dims,
                dropout_rate,
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):
        cfg = super().get_config()
        return cfg


class CustomLayerNormalization(tf.keras.layers.Layer):
    """
    Layer normalization using mean and variance
    gamma and beta are the learnable parameters
    """

    def __init__(self, seq_max_len, embedding_dim, epsilon):
        super(CustomLayerNormalization, self).__init__()
        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.params_shape = (self.seq_max_len, self.embedding_dim)
        g_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=g_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=b_init(shape=self.params_shape, dtype="float32"),
            trainable=True,
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, [-1], keepdims=True)
        normalized = (x - mean) / ((variance + self.epsilon) ** 0.5)
        output = self.gamma * normalized + self.beta
        return output

    def get_config(self):
        cfg = super().get_config()
        return cfg

def embedding(input_seq, item_embedding_layer, positional_embedding_layer):
    seq_embeddings = item_embedding_layer(input_seq)
    positional_seq = tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0)
    positional_seq = tf.tile(positional_seq, [tf.shape(input_seq)[0], 1])
    positional_embeddings = positional_embedding_layer(positional_seq)
    return seq_embeddings, positional_embeddings


def loss_function(y_true, y_pred):
  loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')(y_true, y_pred)
  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)
  return tf.reduce_mean(loss)