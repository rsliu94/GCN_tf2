import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotUniform, Zeros
from tensorflow.keras.regularizers import l2


class GraphConvolution(layers.Layer):
    def __init__(self, units, activation='relu', dropout_rate=0.5,
                 use_bias=True, l2_reg=0, seed=1024):
        super(GraphConvolution, self).__init__()
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.seed = seed

    def build(self, input_shape):
        feature_shape = input_shape[0]
        input_dim = int(feature_shape[-1])
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=GlorotUniform(seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=Zeros(),
                                        name='bias')
        self.dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        self.built = True

    def call(self, inputs, training=None):
        features, A = inputs
        features = self.dropout(features, training=training)
        output = tf.matmul(tf.matmul(A, features), self.kernel)
        if self.use_bias:
            output += self.bias
        if self.activation == 'relu':
            act = tf.nn.relu(output)
        elif self.activation == 'softmax':
            act = tf.nn.softmax(output)
        return act

    def get_config(self):
        config = {'units': self.units,
                  'activation': self.activation,
                  'dropout_rate': self.dropout_rate,
                  'l2_reg': self.l2_reg,
                  'use_bias': self.use_bias,
                  'seed': self.seed
                  }
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GCN(keras.Model):
    def __init__(self, hidden_dim, out_dim, l2_reg=0):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(units=hidden_dim, activation='relu', l2_reg=l2_reg)
        self.gc2 = GraphConvolution(units=out_dim, activation='softmax', l2_reg=l2_reg)

    def call(self, inputs, training=None):
        _, A = inputs
        h1 = self.gc1(inputs)
        h2 = self.gc2([h1, A])
        return h2






