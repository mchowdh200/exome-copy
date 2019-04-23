
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import classification_report

from models import cnn1D_rnn_model
from utils import load_data


X_train, X_val, y_train, y_val = load_data(channels_first=True)

clf = KerasClassifier(build_fn=cnn1D_rnn_model, input_shape=X_train.shape[1:])

callbacks = [EarlyStopping(patience=4, restore_best_weights=True),
             ReduceLROnPlateau(patience=3, factor=0.2)]
clf.fit(X_train, y_train,
        epochs=100,
        batch_size=256,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=callbacks)
# evaluate
target_names = ['non_sv', 'deletion', 'duplication']
y_pred = clf.predict(X_val)
print(classification_report(np.argmax(y_val, axis=1), y_pred,
                            target_names=target_names))
