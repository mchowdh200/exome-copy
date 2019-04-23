
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import normalize
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report

from models import conv1D_model
from utils import load_data

# setup
X_train, X_val, y_train, y_val = load_data()
clf = KerasClassifier(build_fn=conv1D_model, input_shape=X_train.shape[1:])
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
y_pred = clf.predict(X_val)
print(classification_report(np.argmax(y_val, axis=1), y_pred))
