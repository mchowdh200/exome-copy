import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv1D, AveragePooling1D
# from tensorflow.keras.layers import LSTM, Bidirectional
# from tensorflow.keras.layers import Activation, LeakyReLU
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers.experimental import LayerNormalization
# from tensorflow.keras.layers import GaussianNoise, Dropout
# from tensorflow.keras.layers import TimeDistributed, Flatten, Permute
# from tensorflow.keras.layers import multiply
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam


# -----------------------------------------------------------------------------
# Utility Classes

class ModelFactory:
    """
    Helper class used to return a compiled keras model.
    """
    def __init__(self, model_class, **model_params):
        """
        Pass in a keras model class (just the class not an instance)
        """
        self.model_class = model_class
        self.model_params = model_params

    def __call__(self, input_shape=None, lr=1e-3, clipnorm=1, 
                 decay=1e-5, amsgrad=True):
        """
        Factory function to be called by a KerasClassifier object.
        Instantiates a model given by model_class with **kwargs.  
        Compiles model with the rest of the provided arguments.

        Pass the input_shape arg if your model needs to know the input
        shape upon instantiation.
        """

        # yuck
        if input_shape:
            model = self.model_class(input_shape, **self.model_params)
        else:
            model = self.model_class(**self.model_params)

        # TODO If model needs to be run eagerly, then we need to pass in a tensor
        # to evaluate before compiling

        model.compile(
            optimizer= tf.keras.optimizers.Adam(lr=lr, clipnorm=clipnorm,
                                                decay=decay, amsgrad=amsgrad),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model

# -----------------------------------------------------------------------------
# Convolutional Model building blocks
class Conv1DBlock(tf.keras.Model):
    """
    Block containing a conv1d layer followed by dropout, batchnorm, and pooling
    """
    def __init__(self, #input_shape, 
                 filters=128, kernel_size=6,
                 strides=1, dilation_rate=1,
                 dropout_rate=0.2, pool_size=2,
                 data_format='channels_first',
                 normalization_type='batch'):
        super().__init__()
        self.conv1d = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, dilation_rate=dilation_rate,
            data_format=data_format,
            kernel_initializer='glorot_uniform')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        if normalization_type == 'batch':
            self.normalization = tf.keras.layers.BatchNormalization()
        else:
            self.normalization = LayerNormalization(epsilon=1e-6)
        self.leaky_relu = tf.keras.layers.Activation(
            tf.kerasl.layer.LeakyReLU())
        self.avg_pool = tf.keras.layers.AveragePooling1D(
            pool_size=pool_size,
            data_format=data_format)

    def call(self, input_tensor):
        x = self.conv1d(input_tensor)
        x = self.dropout(x)
        x = self.normalization(x)
        x = self.leaky_relu(x)
        return self.avg_pool(x)

    def compute_output_shape(self, input_shape):
        x = self.conv1d.compute_output_shape(input_shape)
        return self.avg_pool.compute_output_shape(x)



# -----------------------------------------------------------------------------
# My custom attention layer
# Take the input and apply a weight matrix and use the outputs as 
# the attention scores.
class AttentionBlock(tf.keras.Model):
    """
    Attention weighting of the input parameterized by a dense 
    feedforward network with softmax output layer.
    """
    def __init__(self, input_shape):
        super(AttentionBlock, self).__init__(name='')
        # self.input_shape = input_shape
        self.permute = tf.keras.layers.Permute((2, 1))
        self.dense = tf.keras.layers.Dense(input_shape[0], activation='relu')
        self.attention_scores = tf.keras.layers.Dense(input_shape[0], 
                                                      name='attention_scores')

    def call(self, input_tensor):
        # expecting input_shape to be (batch_size, time_steps, num_features)
        # format the input to the attention scorer to be 
        # (batch, num_features, time_steps)
        a = self.permute(input_tensor)
        a = self.dense(a)
        a = self.attention_scores(a)
        a = self.permute(a)
        return tf.keras.layers.multiply([a, input_tensor]) # weight input by attention


# -----------------------------------------------------------------------------
# Models
class Conv1DModel(tf.keras.Model):
    def __init__(self, normalization_type='batch'):
        super().__init__()
        self.add_noise = tf.keras.layers.GaussianNoise(stddev=0.01)
        self.conv1 = Conv1DBlock(
            filters=128, 
            kernel_size=12,
            dropout_rate=0.25, 
            pool_size=3,
            normalization_type=normalization_type)
        self.conv2 = Conv1DBlock(
            filters=64, 
            kernel_size=6, 
            dropout_rate=0.25,
            pool_size=3,
            normalization_type=normalization_type)
        self.flatten = tf.keras.layers.Flatten()
        self.softmax = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, input_tensor):
        x = self.add_noise(input_tensor)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.softmax(self.flatten(x))


def rnn_model(input_shape, lr=5e-4, decay=0.0, clipnorm=1, amsgrad=True):
    """
    Simple Bidirectional RNN model, shape of the input is
    (batch_size, seq_length, features)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=256, input_shape=input_shape,
            return_sequences=False)),
        tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            lr=lr,
            clipnorm=clipnorm, 
            decay=decay,
            amsgrad=amsgrad), 
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

def dense_model(input_shape, lr=5e-4, decay=0.0, clipnorm=1, amsgrad=True):
    """
    Simple Bidirectional RNN model, shape of the input is
    (batch_size, seq_length, features)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            lr=lr,
            clipnorm=clipnorm, 
            decay=decay,
            amsgrad=amsgrad), 
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


class Conv1DRNNModel(tf.keras.Model):
    def __init__(self, conv_before=True, conv_after=False):
        super().__init__()

        self.conv_before = conv_before
        self.conv_after = conv_after

        self.add_noise = tf.keras.layers.GaussianNoise(stddev=0.01)

        # pre recurrent conv layers
        self.conv1 = Conv1DBlock(
            # filters=256, 
            filters=128, 
            kernel_size=12,
            dropout_rate=0.25, 
            pool_size=3,
            data_format='channels_last')
        self.conv2 = Conv1DBlock(
            # filters=512, 
            filters=64, 
            kernel_size=6,
            dropout_rate=0.25, 
            pool_size=3,
            data_format='channels_last')

        # recurrent layers
        self.bilstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=256,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
                return_sequences=True))

        self.normalization = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6)

        # if we have conv after, then return sequences
        self.bilstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=256,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
                return_sequences=self.conv_after))

        # post recurrent conv layers
        self.conv3 = Conv1DBlock(
            filters=256,
            kernel_size=6,
            dropout_rate=0.25,
            pool_size=3,
            data_format='channels_last'
        )
        self.flatten = tf.keras.layers.Flatten()

        # output layers
        self.dense = tf.keras.layers.Dense(3, activation='softmax')

    def call(self, input_tensor):
        x = input_tensor
        x = self.add_noise(x)
        if self.conv_before:
            x = self.conv1(x)
            x = self.conv2(x)
        x = self.bilstm1(x)
        x = self.normalization(x)
        x = self.bilstm2(x)
        if self.conv_after:
            x = self.conv3(x)
            x = self.flatten(x)
        return self.dense(x)


class AttentionRNN(tf.keras.Model):
    """
    Bidirectional LSTM augmented with attention layer over each hidden state.
    """
    def __init__(self, input_shape, num_classes=3, 
                 rnn_hidden_size=128, dense_hidden_size=128):
        super().__init__()
        self.num_classes = num_classes
        self.add_noise = tf.keras.layers.GaussianNoise(stddev=0.01)

        # self.project = TimeDistributed(Dense(dense_hidden_size))
        # self.dropout = Dropout(0.2)

        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(rnn_hidden_size, return_sequences=True))
        self.normalization = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6)
        self.attention = AttentionBlock(input_shape)

        self.td_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=dense_hidden_size))
        self.flatten = tf.keras.layers.Flatten()

        self.output_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')

        # used to store attention score for visualization purposes
        # self.attention_score = None

    def call(self, input_tensor):
        x = self.add_noise(input_tensor)
        # this seems to give the model a lot of power, but probably need more
        # data in order to prevent overfitting.
        # x = self.project(x) 
        # x = self.dropout(x)

        x = self.bilstm(x)
        x = self.attention(x)
        x = self.normalization(x)

        x = self.td_dense(x)
        x = self.flatten(x)
        return self.output_layer(x)
