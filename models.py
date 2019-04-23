
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPool1D, AveragePooling1D, 
                                     Flatten, Dense, Dropout, 
                                     BatchNormalization, CuDNNLSTM,
                                     Bidirectional, Activation, LeakyReLU,
                                     GaussianNoise)
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import normalize

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
        # Bidirectional(CuDNNLSTM(units=128,
        #                         kernel_regularizer=l2(1e-4),
        #                         recurrent_regularizer=l2(1e-4),
        #                         return_sequences=True)),
        Bidirectional(CuDNNLSTM(units=128,
                                kernel_regularizer=l2(1e-4),
                                recurrent_regularizer=l2(1e-4),
                                return_sequences=False)),

        ## Output layer -----------------------------------------------------------
        Dense(3, activation='softmax')
        ])

    model.compile(optimizer=Adam(lr=1e-3,
                                 clipnorm=1, 
                                 decay=1e-5,
                                 amsgrad=True), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


