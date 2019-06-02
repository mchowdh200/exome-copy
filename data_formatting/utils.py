import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(del_file='./data/deletions.pkl', 
              dup_file='./data/duplications.pkl', 
              non_file='./data/non_sv.pkl',
              seq_length=500, 
              # test_size=0.1, 
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
    labels = to_categorical(labels)

    # make fixed length sequences
    data_padded = [pad_sequences(d, maxlen=seq_length, 
                                 padding='post',
                                 truncating='post',
                                 dtype='float32')
                   for d in data]
    data_padded = np.array(data_padded)

    if normalize_data:
        data_padded = normalize(data_padded, axis=-1)

    if not channels_first:
        # RNN input shape needs to be (batch_size, seq_length, input_features)
        data_padded = np.swapaxes(data_padded, 1, 2)

    # split data making sure to preserve the class balance
    # return train_test_split(data_padded, labels, 
    #                         stratify=labels, test_size=test_size)
    return data_padded, labels


def examine_data(del_file='./data/deletions.pkl', 
                 dup_file='./data/duplications.pkl', 
                 non_file='./data/non_sv.pkl',
                 seq_length=500, 
                 channels_first=True,
                 normalize_data=True):
    """
    Load data from pickled dataframes.  Combines data into matrices, and
    pads/truncates data, and returns data data in the form of a train test
    split.
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    deletions = pd.read_pickle(del_file)
    # duplications = pd.read_pickle(dup_file)
    # non_sv = pd.read_pickle(non_file)

    # print(deletions.data.values.shape)
    # print(duplications.data.values.shape)
    # print(non_sv.data.values.shape)

    deletions_padded = np.array([pad_sequences(d, maxlen=seq_length,
                                      padding='post',
                                      truncating='post',
                                      dtype='float32')
                        for d in deletions.data.values])

    if normalize_data:
        min_val = np.min(deletions_padded, axis=0, keepdims=True)
        # deletions_padded = normalize(deletions_padded, axis=2)
        deletions_padded = deletions_padded/(np.max(deletions_padded+1, axis=0, keepdims=True))


    # plot a random piece of data
    i = np.random.randint(0, len(deletions.values))
    sns.heatmap(deletions_padded[100])
    # sns.heatmap(non_sv.data.values[14])
    plt.savefig('test.png')



    # data = np.concatenate((
    #     non_sv.data.values,
    #     deletions.data.values,
    #     duplications.data.values
    # ))

if __name__ == '__main__':
    examine_data()



