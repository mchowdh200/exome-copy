
import numpy as np
# import pandas as pd
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.utils import normalize
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report

from models import rnn_model
from utils import load_data


# data setup
X_train, X_val, y_train, y_val = load_data(channels_first=False)

# model setup
# model = Sequential([
#     Bidirectional(CuDNNLSTM(units=256, input_shape=X_train.shape[1:])),
#     Dense(units=3, activation='softmax')
# ])

# model.compile(optimizer=Adam(lr=5e-4,
#                              clipnorm=1, 
#                              # decay=0.05,
#                              amsgrad=True), 
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
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
