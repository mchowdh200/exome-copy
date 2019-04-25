
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.utils import normalize
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from models import rnn_model
from utils import load_data


# data setup
X_train, X_val, y_train, y_val = load_data(channels_first=False)

clf = KerasClassifier(build_fn=rnn_model, input_shape=X_train.shape[1:])

callbacks = [EarlyStopping(patience=4),
             ReduceLROnPlateau(patience=3, factor=0.2)]

# train
clf.fit(X_train, y_train,
        epochs=20,
        batch_size=256,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=callbacks)

# evaluate
y_pred = clf.predict(X_val)
print(classification_report(np.argmax(y_val, axis=1), y_pred))
