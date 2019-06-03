import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, AveragePooling1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers.experimental import LayerNormalization
from tensorflow.keras.layers import GaussianNoise, Dropout
from tensorflow.keras.layers import TimeDistributed, Flatten, Permute
from tensorflow.keras.layers import multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


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

        model.compile(optimizer=Adam(lr=lr, clipnorm=clipnorm,
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
        self.conv1d = Conv1D(filters=filters, kernel_size=kernel_size,
                             strides=strides, dilation_rate=dilation_rate,
                             data_format=data_format,
                             kernel_initializer='glorot_uniform')
        self.dropout = Dropout(rate=dropout_rate)
        if normalization_type == 'batch':
            self.normalization = BatchNormalization()
        else:
            self.normalization = LayerNormalization(epsilon=1e-6)
        self.leaky_relu = Activation(LeakyReLU())
        self.avg_pool = AveragePooling1D(pool_size=pool_size,
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
        self.permute = Permute((2, 1))
        self.dense = Dense(input_shape[0], activation='relu')
        self.attention_scores = Dense(input_shape[0], name='attention_scores')

    def call(self, input_tensor):
        # expecting input_shape to be (batch_size, time_steps, num_features)
        # format the input to the attention scorer to be 
        # (batch, num_features, time_steps)
        a = self.permute(input_tensor)
        a = self.dense(a)
        a = self.attention_scores(a)
        a = self.permute(a)
        return multiply([a, input_tensor]) # weight input by attention


# -----------------------------------------------------------------------------
# Models
class Conv1DModel(tf.keras.Model):
    def __init__(self, normalization_type='batch'):
        super().__init__()
        self.add_noise = GaussianNoise(stddev=0.01)
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
        self.flatten = Flatten()
        self.softmax = Dense(3, activation='softmax')

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
    model = Sequential([
        Bidirectional(LSTM(units=256, input_shape=input_shape,
                           return_sequences=False)),
        LayerNormalization(epsilon=1e-6),
        # Bidirectional(LSTM(units=256, input_shape=input_shape,)),
        Dense(units=3, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(lr=lr,
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
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(units=3, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(lr=lr,
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

        self.add_noise = GaussianNoise(stddev=0.01)

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
        self.bilstm1 = Bidirectional(
            LSTM(units=256,
                 kernel_regularizer=l2(1e-4),
                 recurrent_regularizer=l2(1e-4),
                 return_sequences=True))

        self.normalization = LayerNormalization(epsilon=1e-6)

        # if we have conv after, then return sequences
        self.bilstm2 = Bidirectional(
            LSTM(units=256,
                 kernel_regularizer=l2(1e-4),
                 recurrent_regularizer=l2(1e-4),
                 return_sequences=self.conv_after))

        # post recurrent conv layers
        self.conv3 = Conv1DBlock(
            filters=256,
            kernel_size=6,
            dropout_rate=0.25,
            pool_size=3,
            data_format='channels_last'
        )
        self.flatten = Flatten()

        # output layers
        self.dense = Dense(3, activation='softmax')

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
        self.add_noise = GaussianNoise(stddev=0.01)

        # self.project = TimeDistributed(Dense(dense_hidden_size))
        # self.dropout = Dropout(0.2)

        self.bilstm = Bidirectional(LSTM(rnn_hidden_size, return_sequences=True))
                       
        self.normalization = LayerNormalization(epsilon=1e-6)
        self.attention = AttentionBlock(input_shape)

        self.td_dense = TimeDistributed(Dense(units=dense_hidden_size))
        self.flatten = Flatten()

        self.output_layer = Dense(units=num_classes, activation='softmax')

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
