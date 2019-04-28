
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import classification_report

from models import Conv1DRNNModel, ModelFactory
from utils import load_data

tf.compat.v1.disable_eager_execution()

X_train, X_val, y_train, y_val = load_data(channels_first=False)

# clf = KerasClassifier(build_fn=cnn1D_rnn_model, input_shape=X_train.shape[1:])
clf = KerasClassifier(build_fn=ModelFactory(Conv1DRNNModel, 
                                            conv_before=True,
                                            conv_after=False), 
                      lr=2e-3,
                      decay=1e-5,)

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
y_pred = clf.model.predict(X_val)
print(classification_report(np.argmax(y_val, axis=1), 
                            np.argmax(y_pred, axis=1),
                            target_names=target_names))
