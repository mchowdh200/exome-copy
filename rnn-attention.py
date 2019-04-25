import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.utils import normalize
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from models import ModelFactory, AttentionRNN
from utils import load_data

from tensorflow.keras.optimizers import Adam


# data setup
X_train, X_val, y_train, y_val = load_data(channels_first=False, normalize_data=True)

clf = KerasClassifier(build_fn=ModelFactory(AttentionRNN), 
                      input_shape=X_train.shape[1:],
                      lr=2.5e-3,)

callbacks = [EarlyStopping(patience=4, restore_best_weights=False),
             ReduceLROnPlateau(patience=3, factor=0.2)]

# train
clf.fit(X_train, y_train,
        epochs=50,
        batch_size=256,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=callbacks)

# evaluate
y_pred = clf.model.predict(X_val)
print(y_pred.shape)
print(classification_report(np.argmax(y_val, axis=1), 
                            np.argmax(y_pred, axis=1)))
