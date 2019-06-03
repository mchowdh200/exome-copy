import numpy as np
import pandas as pd
import tensorflow as tf

def load_data(del_file='./data/dataset/deletions.pkl', 
              dup_file='./data/dataset/duplications.pkl', 
              non_file='./data/dataset/non_sv.pkl',
              seq_length=500, 
              channels_first=True,
              normalize_data=False):
    """
    Load data from pickled dataframes.  Combines data into matrices, and
    pads/truncates data, and returns data data in the form of a train test
    split.
    """
    deletions = pd.read_pickle(del_file)
    duplications = pd.read_pickle(dup_file)
    non_sv = pd.read_pickle(non_file)

    # combine data and create labels
    data = np.concatenate((
        non_sv.data.values,
        deletions.data.values,
        duplications.data.values
    ))

    labels = np.concatenate((
        np.zeros((len(non_sv),)),
        np.full((len(deletions,)), fill_value=1),
        np.full((len(duplications,)), fill_value=2)
    ))
    labels = tf.keras.utils.to_categorical(labels)

    # make fixed length sequences
    data_padded = [
        tf.keras.preprocessing.sequence.pad_sequences(
            d, maxlen=seq_length, 
            padding='post',
            truncating='post',
            dtype='float32')
        for d in data]
    data_padded = np.array(data_padded)

    if normalize_data:
        data_padded = tf.keras.utils.normalize(data_padded)

    if not channels_first:
        # RNN input shape needs to be (batch_size, seq_length, input_features)
        data_padded = np.swapaxes(data_padded, 1, 2)

    return data_padded, labels


