
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (average_precision_score, classification_report, 
                             confusion_matrix, precision_recall_curve, 
                             auc, roc_curve, precision_recall_fscore_support)
from sklearn.utils import shuffle
from models import *
from utils import *

# tf.compat.v1.disable_eager_execution()


def run_model(data, labels, model_factory, callbacks, compile_params,
              classes, folds=10, n_classes=3, epochs=100, batch_size=512,
              out_dir='', model_name='model'):

    # data, labels = load_data()
    ratios = np.array([np.sum(np.argmax(labels, axis=1) == 0),
              np.sum(np.argmax(labels, axis=1) == 1),
              np.sum(np.argmax(labels, axis=1) == 2),
             ])/len(labels)
    classes = {0: 'Non-SV', 1: 'Deletion', 2: 'Duplication'}

    # used to make shape of tpr/fpr, etc the same
    x = np.linspace(0, 1, 1000) 
    # used for ROC curves
    tprs = np.zeros((folds, n_classes, 1000))
    roc_aucs = np.zeros((folds, n_classes))

    # used for pr curves
    precisions = np.zeros((folds, n_classes, 1000))
    # recalls = np.zeros((folds, n_classes, x.shape[2]))
    pr_aucs = np.zeros((folds, n_classes))

    # used for precision/recal/F1 metrics
    # axis 1 is for precision, recall, f1, and support respectively
    pr_F1 = np.zeros((folds, 4, n_classes))

    # cross val loop ----------------------------------------------------------------------------------
    for i, (train, test) in enumerate(StratifiedKFold(n_splits=folds, shuffle=True)
                                      .split(data, np.argmax(labels, axis=1))):
        print('-'*100)
        print('FOLD {}'.format(i))
        print('-'*100)

        clf = KerasClassifier(
            build_fn=model_factory,
            **compile_params)
        
        X, y = shuffle(data[train], labels[train])
        clf.fit(X, y,
                # epochs=100,
                epochs=epochs,
                batch_size=batch_size,
                verbose=2,
                validation_split=0.1,
                callbacks=callbacks)

        y_pred = clf.model.predict(data[test])

        # ROC/PR curves for each class
        for j in range(n_classes):
            fpr, tpr, _ = roc_curve(labels[test][:, j], y_pred[:, j])
            tprs[i, j, :] = np.interp(x, fpr, tpr)
            roc_aucs[i, j] = auc(x, tprs[i, j])
            precision, recall, _ = precision_recall_curve(labels[test][:, j], y_pred[:, j])
            precisions[i, j, :] = np.interp(x, precision, recall)
            pr_aucs[i, j] = average_precision_score(labels[test][:, j], y_pred[:, j])

        # Precision/Recall/F1 for each class
        pr_F1[i, :, :] = precision_recall_fscore_support(np.argmax(labels[test], axis=1),
                                                         np.argmax(y_pred, axis=1),
                                                         average=None)

    # get mean/stddev of our metrics ------------------------------------------------------------------
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
        print('%d\t%.3f +/- %.3f\t%.3f +/- %.3f\t%.3f +/- %.3f\t%d' 
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
