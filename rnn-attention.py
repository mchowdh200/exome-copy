import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from models import ModelFactory, AttentionRNN
from utils import load_data


tf.compat.v1.disable_eager_execution()

# data setup
X_train, X_val, y_train, y_val = load_data(channels_first=False, normalize_data=True)

clf = KerasClassifier(build_fn=ModelFactory(AttentionRNN, rnn_hidden_size=256, dense_hidden_size=256), 
                      input_shape=X_train.shape[1:],
                      lr=1e-3,)

callbacks = [EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
             ReduceLROnPlateau(patience=3, factor=0.2, monitor='val_accuracy')]

# train
clf.fit(X_train, y_train,
        epochs=100,
        batch_size=512,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=callbacks)

# evaluate
y_pred = clf.model.predict(X_val)
print(y_pred.shape)
print(classification_report(np.argmax(y_val, axis=1), 
                            np.argmax(y_pred, axis=1)))
