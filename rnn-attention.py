import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle
from models import ModelFactory, AttentionRNN
from utils import load_data

tf.compat.v1.disable_eager_execution()

# data setup
data, labels = load_data(channels_first=False, normalize_data=True)
data, labels = shuffle(data, labels)

# model setup
get_model = ModelFactory(AttentionRNN, rnn_hidden_size=32, dense_hidden_size=32)
model = get_model(input_shape=data.shape[1:], lr=1e-3,)

callbacks = [EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
             ReduceLROnPlateau(patience=3, factor=0.2, monitor='val_accuracy')]


# train
# model.build((None, data.shape[1], data.shape[2]))

model.fit(
    data, labels,
    epochs=1,
    batch_size=256,
    verbose=1,
    validation_split=0.1,
    callbacks=callbacks)

model(np.expand_dims(data[1000], axis=0))
x = model.attention_score.numpy()
x = np.swapaxes(x, 0, 1)

fig = sns.heatmap(x)
plt.xlabel('hidden state')
plt.ylabel('actviation dimension')
plt.savefig('heatmap_nosoftmax.png')

# evaluate
# y_pred = model.predict(X_val)
# print(y_pred.shape)
# print(classification_report(np.argmax(y_val, axis=1), 
#                             np.argmax(y_pred, axis=1)))
