import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, AveragePooling1D
from tensorflow.keras.layers import CuDNNLSTM, Bidirectional
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, GaussianNoise, Dropout
from tensorflow.keras.layers import TimeDistributed, Flatten, Permute
from tensorflow.keras.layers import Input, multiply

from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import normalize
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


class ModelFactory:
    def __init__(self, model_class):
        """
        Pass in a keras model class (just the class not an instance)
        """
        self.model_class = model_class

    def __call__(self, input_shape, 
                 lr=1e-3, clipnorm=1, decay=1e-5, amsgrad=True,
                 **kwargs):
        """
        Factory function to be called by a KerasClassifier object
        """
        model = self.model_class(input_shape, **kwargs)
        model.compile(optimizer=Adam(lr=lr, clipnorm=clipnorm,
                                     decay=decay,amsgrad=amsgrad),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model




# Model Factories ------------------------------------------------------------
def conv1D_model(input_shape):
    """
    Simple 1d convnet. Input shape is (batch_size, num_channels, sequence_length)
    """
    # TODO add more input parameters
    model = Sequential([
        Conv1D(input_shape=input_shape,
               filters=128,
               kernel_size=12,
               strides=1,
               dilation_rate=1,
               data_format='channels_first',
               kernel_initializer='glorot_uniform'),
        Dropout(0.25),
        BatchNormalization(),
        Activation(LeakyReLU()),
        MaxPool1D(pool_size=3),

        Conv1D(filters=64,
               kernel_size=6,
               strides=1,
               dilation_rate=1,
               data_format='channels_first',
               kernel_initializer='glorot_uniform'),
        BatchNormalization(),
        Activation(LeakyReLU()),
        Dropout(0.25),
        MaxPool1D(pool_size=3),

        Flatten(), 
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer=Adam(clipnorm=1, amsgrad=True), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def rnn_model(input_shape):
    """
    Simple Bidirectional RNN model, shape of the input is
    (batch_size, seq_length, features)
    """
    model = Sequential([
        Bidirectional(CuDNNLSTM(units=256, input_shape=input_shape)),
        Dense(units=3, activation='softmax')
    ])

    model.compile(optimizer=Adam(lr=5e-4,
                                 clipnorm=1, 
                                 # decay=0.05,
                                 amsgrad=True), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def cnn1D_rnn_model(input_shape):
    """
    Initial layers consist of a 1d convnet.  The output feature maps are then
    fed into a bidirectional RNN.  Input shape is 
    (batch_size, num_channels, seq_length)
    """
    model = Sequential([
        ## Conv layers -----------------------------------------------------------
        GaussianNoise(stddev=0.05, input_shape=input_shape),
        Conv1D(filters=128,
               kernel_size=12,
               strides=1,
               dilation_rate=1,
               data_format='channels_first',
               kernel_initializer='glorot_uniform'),
        Dropout(0.25),
        BatchNormalization(),
        Activation(LeakyReLU()),
        # MaxPool1D(pool_size=3, data_format='channels_first'),
        AveragePooling1D(pool_size=3, data_format='channels_first'),

        Conv1D(filters=256,
               kernel_size=6,
               strides=1,
               dilation_rate=1,
               data_format='channels_first',
               kernel_initializer='glorot_uniform'),
        BatchNormalization(),
        Activation(LeakyReLU()),
        Dropout(0.25),
        MaxPool1D(pool_size=3, data_format='channels_first'),
        AveragePooling1D(pool_size=3, data_format='channels_first'),

        ## recurrent layers -------------------------------------------------------
        Bidirectional(CuDNNLSTM(units=128,
                                kernel_regularizer=l2(1e-4),
                                recurrent_regularizer=l2(1e-4),
                                return_sequences=True)),
        Bidirectional(CuDNNLSTM(units=128,
                                kernel_regularizer=l2(1e-4),
                                recurrent_regularizer=l2(1e-4),
                                return_sequences=False)),

        ## Output layer -----------------------------------------------------------
        Dense(3, activation='softmax')
        ])

    model.compile(optimizer=Adam(lr=1e-3, clipnorm=1, decay=1e-5, amsgrad=True), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# def attention_augmented_rnn(input_shape):
#     _input = Input(shape=input_shape)
#     x = GaussianNoise(stddev=0.05)(_input)
#     x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
#     x = AttentionBlock(input_shape)(x)
#     x = TimeDistributed(Dense(units=128, activation=LeakyReLU()))(x)
#     x = Flatten()(x)
#     output = Dense(units=3, activation='softmax')(x)

#     model = Model(inputs=_input, outputs=output)
#     model.compile(optimizer=Adam(lr=1e-3, clipnorm=1, decay=1e-5, amsgrad=True), 
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model



class AttentionBlock(tf.keras.Model):
    def __init__(self, input_shape):
        super(AttentionBlock, self).__init__(name='')
        # self.input_shape = input_shape
        self.permute = Permute((2, 1))
        # self.attention_scores = Dense(input_shape[0], activation='softmax')
        self.attention_scores = Dense(input_shape[0])

    def call(self, input_tensor):
        # expecting input_shape to be (batch_size, time_steps, num_features)
        time_steps, num_features = input_tensor.shape[1:]

        # format the input to the attention scorer to be 
        # (batch, num_features, time_steps)
        a = self.permute(input_tensor)
        a = self.attention_scores(a)
        a = self.permute(a)
        return multiply([a, input_tensor]) # weight input by attention

class AttentionRNN(tf.keras.Model):
    """
    Bidirectional LSTM augmented with attention layer over each hidden state.
    """
    def __init__(self, input_shape, num_classes=3, 
                 rnn_hidden_size=128, dense_hidden_size=128):
        super().__init__()
        self.num_classes = num_classes
        # self.input_layer = Input(shape=input_shape)
        self.add_noise = GaussianNoise(stddev=0.05)
        self.bilstm = Bidirectional(CuDNNLSTM(units=rnn_hidden_size,
                                            return_sequences=True))
        self.attention = AttentionBlock(input_shape)
        self.td_dense = TimeDistributed(Dense(units=dense_hidden_size))
        self.flatten = Flatten()
        self.output_layer = Dense(units=num_classes, activation='softmax')

    def call(self, input_tensor):
        # _input = self.input_layer(input_tensor)
        x = self.add_noise(input_tensor)
        x = self.bilstm(x)
        x = self.attention(x)
        x = self.td_dense(x)
        x = self.flatten(x)
        return self.output_layer(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)
