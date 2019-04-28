
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import normalize
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report

from models import ModelFactory, Conv1DModel
from utils import load_data
tf.compat.v1.disable_eager_execution()

# setup
X_train, X_val, y_train, y_val = load_data()

clf = KerasClassifier(build_fn=ModelFactory(Conv1DModel, normalization_type='batch'),
                      lr=2.5e-3)
callbacks = [EarlyStopping(patience=4, restore_best_weights=True),
             ReduceLROnPlateau(patience=3, factor=0.2)]

# train
clf.fit(X_train, y_train,
        epochs=20,
        batch_size=512,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=callbacks)

# evaluate
y_pred = clf.model.predict(X_val)
print(classification_report(np.argmax(y_val, axis=1), 
                            np.argmax(y_pred, axis=1)))
