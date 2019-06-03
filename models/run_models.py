import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seq-length', dest='seq_length', nargs='?', type=int, default=500,
                    help='Length of input sequences.  Default = 500.')
parser.add_argument('--folds', dest='folds', nargs='?', type=int, default=10,
                    help='Number of folds for cross validation.  Default = 10.')
parser.add_argument('--batch-size', dest='batch_size', nargs='?', type=int, default=512,
                    help='Training batch size.  Default = 512.')
parser.add_argument('--epochs', dest='epochs', nargs='?', type=int, default=100,
                    help='Epochs per fold.  Default = 100.')
parser.add_argument('--out-dir', dest='out_dir', nargs='?', type=str, default='output',
                    help='Output directory.  Default is output.')
parser.add_argument('--model-type', dest='model_name', nargs='?', type=str, default='CNN',
                    help="""Model to run.  Choices are CNN, RNN, CNN-RNN, RNN-Attention.
                    Default is CNN.""")
parser.add_argument('--lr', nargs='?', dest='lr', type=float, default=1e-3,
                    help='Optimizer learning rate.  Default is 1e-3.')
parser.add_argument('--decay', nargs='?', dest='decay', type=float, default=0.0,
                    help='Optimizer learning rate decay.  Default is 0.')
parser.add_argument('--patience', dest='patience', nargs='?', type=int, default=6,
                    help='Early stopping patience.  Default is 6.')
parser.add_argument('--verbose', dest='verbose', nargs='?', type=int, default=2,
                    help='verbose argument passed into the model fit method. Default is 2')
args = parser.parse_args()

print(args)

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models import *
from utils import load_data
from cross_val import run_model
tf.compat.v1.disable_eager_execution()



# get dataset
channels_first = True if args.model_name == 'CNN' else False
data, labels = load_data(seq_length=args.seq_length, 
                         channels_first=channels_first, normalize_data=True)

# these paramters will be passed to the KerasClassifier 
# upon instantiation of our model.
compile_params = dict(lr=args.lr, decay=args.decay)
if args.model_name in ('RNN', 'RNN-Attention', 'Dense'):
    compile_params['input_shape'] = data.shape[1:]

callbacks = [EarlyStopping(monitor='val_accuracy',
                           patience=args.patience, 
                           restore_best_weights=True),
             ReduceLROnPlateau(monitor='val_accuracy',
                               patience=max(args.patience-2, 1), 
                               factor=0.2)]

classes = {0: 'Non-SV', 1: 'Deletion', 2: 'Duplication'}
class_weight = {0: 1.0, 1: 1.0, 2: 2.0}

model_factory = {
    'CNN': ModelFactory(
        Conv1DModel, 
        normalization_type='batch'),

    'RNN': rnn_model,

    'CNN-RNN': ModelFactory(
        Conv1DRNNModel, 
        conv_before=True, 
        conv_after=False),

    'RNN-Attention': ModelFactory(
        AttentionRNN, 
        rnn_hidden_size=256,
        dense_hidden_size=256),
    'Dense': ModelFactory(
        dense_model
    )
}

run_model(
    data, labels,
    model_factory=model_factory[args.model_name],
    callbacks=callbacks,
    compile_params=compile_params,
    classes=classes,
    folds=args.folds,
    n_classes=3,
    epochs=args.epochs,
    batch_size=args.batch_size,
    out_dir=args.out_dir.rstrip()+'/',
    model_name=args.model_name,
    verbose=args.verbose,
    class_weight=class_weight
)

