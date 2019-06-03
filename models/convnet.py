import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models import ModelFactory, Conv1DModel
from utils import load_data
from cross_val import run_model
tf.compat.v1.disable_eager_execution()

# setup
data, labels = load_data()
classes = {0: 'Non-SV', 1: 'Deletion', 2: 'Duplication'}

model_factory = ModelFactory(Conv1DModel, normalization_type='batch')
callbacks = [EarlyStopping(patience=4, restore_best_weights=True),
             ReduceLROnPlateau(patience=3, factor=0.2)]
compile_params = dict(lr=1e-3)

run_model(
    data, labels,
    model_factory=model_factory,
    callbacks=callbacks,
    compile_params=compile_params,
    classes=classes,
    folds=1,
    n_classes=3,
    epochs=1,
    batch_size=512,
    out_dir='output/',
    model_name='Conv1D',
    verbose=1
)
