import os
import argparse
import numpy as np
import sklearn.utils
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import models
from utils import load_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def run_model(X_train, y_train, X_val, y_val, X_test, y_test,
              model_factory, callbacks, compile_params,
              classes, n_fits=10, n_classes=3, 
              class_weight=None, epochs=100, batch_size=512,
              out_dir='', model_name='model', verbose=2):
    """
    INPUTS:
        - X_*, y_*: Data and label matrices for train/val/test sets

        - model_factory: function that returns a compiled keras model to be
          passed to the 

        - callbacks: list of keras callbacks passed to model fit method

        - compile_params: dict of parameters passed to KerasClassifier **skparams

        - classes: dictionary mapping numeric label to class

        - n_fits: number of times to fit model
        
        - n_classes: number of posible class labels

        - class_weight: dictionary used to weight loss function by class

        - epochs: max number of epochs to fit for

        - batch_size: num of examples per iteration in gradient descent

        - outdir: directory to save results to

        - model_name: string prepended to result files

        - verbose: how much information for the fit method to display
    """

    # used for baselines in PR curves
    ratios = np.array([np.sum(np.argmax(y_test, axis=1) == 0),
                       np.sum(np.argmax(y_test, axis=1) == 1),
                       np.sum(np.argmax(y_test, axis=1) == 2),
                      ])/len(y_test)
    # classes = {0: 'Non-SV', 1: 'Deletion', 2: 'Duplication'}

    # used to make shape of tpr/fpr, etc the same
    x = np.linspace(0, 1, 1000) 
    # used for ROC curves
    tprs = np.zeros((n_fits, n_classes, 1000))
    roc_aucs = np.zeros((n_fits, n_classes))

    # used for pr curves
    precisions = np.zeros((n_fits, n_classes, 1000))
    # recalls = np.zeros((n_fits, n_classes, x.shape[2]))
    pr_aucs = np.zeros((n_fits, n_classes))

    # used for precision/recal/F1 metrics
    # axis 1 is for precision, recall, f1, and support respectively
    pr_F1 = np.zeros((n_fits, 4, n_classes))


    # train model ----------------------------------------------------------------------------------
    for i in range(n_fits):
        print('-'*100)
        print('FOLD {}'.format(i))
        print('-'*100)

        clf = tf.keras.wrappers.scikit_learn.KerasClassifier( 
            build_fn=model_factory, 
            **compile_params)
        
        X, y = sklearn.utils.shuffle(X_train, y_train)
        clf.fit(
            X, y,
            class_weight=class_weight,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=(X_val, y_val),
            callbacks=callbacks)

        y_pred = clf.model.predict(X_test)

        # ROC/PR curves for each class
        for j in range(n_classes):
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_test[:, j], y_pred[:, j])
            tprs[i, j, :] = np.interp(x, fpr, tpr)
            roc_aucs[i, j] = sklearn.metrics.auc(x, tprs[i, j])
            precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test[:, j], y_pred[:, j])
            precisions[i, j, :] = np.interp(x, precision, recall)
            pr_aucs[i, j] = sklearn.metrics.average_precision_score(y_test[:, j], y_pred[:, j])

        # Precision/Recall/F1 for each class
        pr_F1[i, :, :] = sklearn.metrics.precision_recall_fscore_support(np.argmax(y_test, axis=1),
                                                                         np.argmax(y_pred, axis=1),
                                                                         average=None)

    # get mean/stddev of our metrics ---------------------------------------------------------------
    avg_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    avg_roc_auc = np.mean(roc_aucs, axis=0)
    std_roc_auc = np.std(roc_aucs, axis=0)
    avg_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    avg_pr_auc = np.mean(pr_aucs, axis=0)
    std_pr_auc = np.std(pr_aucs, axis=0)
    avg_prf1 = np.mean(pr_F1, axis=0)
    std_prf1 = np.std(pr_F1, axis=0)

    # Classification report with mean/std metrics -----------------------------------------------------
    print('\tprec\t\trecall\t\tf1\t\tsupport\n')
    for i in range(n_classes):
        print('%d\t%.3f (+/- %.3f)\t%.3f (+/- %.3f)\t%.3f (+/- %.3f)\t%d' 
              % (i, avg_prf1[0][i], std_prf1[0][i],
                 avg_prf1[1][i], std_prf1[1][i],
                 avg_prf1[2][i], std_prf1[2][i],
                 avg_prf1[3][i]))

    # Plot ROC/PR Curves ------------------------------------------------------------------------------
    for i in range(n_classes):
        f = plt.figure(figsize=(8, 8))
        plt.plot(x, x, color='k', linestyle='dashed', label='baseline')
        plt.plot(x, avg_tpr[i], color='k', label='Mean ROC')
        plt.fill_between(x, 
                         np.minimum(avg_tpr[i] + std_tpr[i], 1),
                         np.maximum(avg_tpr[i] - std_tpr[i], 0),
                         color='grey', alpha=0.2, 
                         label=r'$\pm$ 1 standard deviation'
                        )
        plt.title(model_name + r'Mean ROC: %s AUC = %.2f $\pm$ %0.2f' % (classes[i], avg_roc_auc[i], std_roc_auc[i]))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim((0, 1))
        plt.ylim((0, 1.05))
        plt.legend()
        f.savefig(out_dir + '{0}.{1}.roc.png'.format(model_name, classes[i]))

        f = plt.figure(figsize=(8, 8))
        plt.plot(x, np.ones_like(x)*ratios[i], color='k', linestyle='dashed', label='baseline')
        plt.plot(x, avg_precision[i], color='k', label='Mean PRC')
        plt.fill_between(x,
                         np.minimum(avg_precision[i] + std_precision[i], 1),
                         np.maximum(avg_precision[i] - std_precision[i], 0),
                         color='grey', alpha=0.2, 
                         label=r'$\pm$ 1 standard deviation'
                        )
        plt.title(model_name + r'Mean Precision/Recall Curve: %s AUC = %.2f $\pm$ %0.3f' % (classes[i], avg_pr_auc[i], std_pr_auc[i]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim((0, 1))
        plt.ylim((0, 1.05))
        plt.legend()
        f.savefig(out_dir + '{0}.{1}.pr.png'.format(model_name, classes[i]))

def model_setup(args):
    # get dataset
    channels_first = args.model_name == 'CNN'

    normalize_data = True
    X_train, y_train = load_data(
        '../data/DataFrames/train/train_del.pkl',
        '../data/DataFrames/train/train_dup.pkl',
        '../data/DataFrames/train/train_nosv.pkl',
        seq_length=args.seq_length, 
        channels_first=channels_first, 
        normalize_data=normalize_data)
    X_val, y_val = load_data(
        '../data/DataFrames/val/val_del.pkl',
        '../data/DataFrames/val/val_dup.pkl',
        '../data/DataFrames/val/val_nosv.pkl',
        seq_length=args.seq_length, 
        channels_first=channels_first, 
        normalize_data=normalize_data)
    X_test, y_test = load_data(
        '../data/DataFrames/test/test_del.pkl',
        '../data/DataFrames/test/test_dup.pkl',
        '../data/DataFrames/test/test_nosv.pkl',
        seq_length=args.seq_length, 
        channels_first=channels_first, 
        normalize_data=normalize_data)

    # these paramters will be passed to the KerasClassifier 
    # upon instantiation of our model.
    compile_params = dict(lr=args.lr, decay=args.decay)
    if args.model_name in ('RNN', 'RNN-Attention', 'Dense'):
        compile_params['input_shape'] = X_train.shape[1:]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=args.patience, 
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=max(args.patience-2, 1), 
            factor=0.2)]

    classes = {0: 'Non-SV', 1: 'Deletion', 2: 'Duplication'}
    class_weight = {0: 1.0, 1: 1.0, 2: 2.0}

    model_factory = {
        'CNN': models.ModelFactory(
            models.Conv1DModel, 
            normalization_type='batch'),

        'RNN': models.rnn_model,

        'CNN-RNN': models.ModelFactory(
            models.Conv1DRNNModel, 
            conv_before=True, 
            conv_after=False),

        'RNN-Attention': models.ModelFactory(
            models.AttentionRNN, 
            rnn_hidden_size=256,
            dense_hidden_size=256),
        'Dense': models.ModelFactory(
            models.dense_model
        )
    }

    run_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_factory=model_factory[args.model_name],
        callbacks=callbacks,
        compile_params=compile_params,
        classes=classes,
        n_fits=args.n_fits,
        n_classes=3,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_dir=args.out_dir.rstrip()+'/',
        model_name=args.model_name,
        verbose=args.verbose,
        class_weight=class_weight
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seq-length', dest='seq_length', nargs='?', type=int, default=500,
        help='Length of input sequences.  Default = 500.')
    parser.add_argument(
        '--n-fits', dest='n_fits', nargs='?', type=int, default=10,
        help='Number of times to fit the model.  Default = 10.')
    parser.add_argument(
        '--batch-size', dest='batch_size', nargs='?', type=int, default=512,
        help='Training batch size.  Default = 512.')
    parser.add_argument(
        '--epochs', dest='epochs', nargs='?', type=int, default=100,
        help='Epochs per fold.  Default = 100.')
    parser.add_argument(
        '--out-dir', dest='out_dir', nargs='?', type=str, default='../output',
        help='Output directory.  Default is ../output.')
    parser.add_argument(
        '--model-type', dest='model_name', nargs='?', type=str, default='CNN',
        help="""Model to run.  Choices are CNN, RNN, CNN-RNN, RNN-Attention.
        Default is CNN.""")
    parser.add_argument(
        '--lr', nargs='?', dest='lr', type=float, default=1e-3,
        help='Optimizer learning rate.  Default is 1e-3.')
    parser.add_argument(
        '--decay', nargs='?', dest='decay', type=float, default=0.0,
        help='Optimizer learning rate decay.  Default is 0.')
    parser.add_argument(
        '--patience', dest='patience', nargs='?', type=int, default=6,
        help='Early stopping patience.  Default is 6.')
    parser.add_argument(
        '--verbose', dest='verbose', nargs='?', type=int, default=2,
        help='verbose argument passed into the model fit method. Default is 2')
    args = parser.parse_args()

    print(args)
    model_setup(args)
