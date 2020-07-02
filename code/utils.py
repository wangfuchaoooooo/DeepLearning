import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

matplotlib.use('agg')


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return directory_path
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if y_true_val is not None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def compute_metrics_(y_true, y_pred):
    res = pd.DataFrame(data=np.zeros((1, 3), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_hist(output_directory, hist, lr=True):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(os.path.join(output_directory, 'history.csv'), index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    if lr is True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(os.path.join(output_directory, 'df_best_model.csv'), index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code
    # plot losses
    plot_epochs_metric(hist, os.path.join(output_directory, 'epochs_loss.png'))


def save_metrics(output_directory, y_pred, y_true, duration, y_true_val=None, y_pred_val=None):
    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(os.path.join(output_directory, 'df_metrics.csv'), index=False)

    return df_metrics


def compute_metrics(output_directory, y_pred, y_true):
    df_metrics = compute_metrics_(y_true, y_pred)
    df_metrics.to_csv(os.path.join(output_directory, 'df_metrics.csv'), index=False)
    return df_metrics


def save_confusion(true_label, pred_label, classes, save_path='/'):
    lmr_matrix = confusion_matrix(true_label, pred_label)
    acc_score = accuracy_score(true_label, pred_label)

    plt.imshow(lmr_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Pre label')
    plt.ylabel('True label')
    for i, j in itertools.product(range(lmr_matrix.shape[0]), range(lmr_matrix.shape[1])):
        plt.text(j, i, lmr_matrix[i, j])
    plt.title('confusion matrix acc={:.3f}'.format(acc_score), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion.png'))
    plt.cla()


def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def load_data(data_name):
    if data_name == 'mnist':
        from dataSet import load_mnist_data
        x_train, y_train, x_test, y_test = \
            load_mnist_data.read_data('./dataSet/data/mnist/mnist.npz')
        return x_train, y_train, x_test, y_test
    elif data_name == 'cifar':
        from dataSet import load_cifar_data
        x_train, y_train, x_test, y_test = \
            load_cifar_data.read_data('./data/cifar10_data')
        return x_train, y_train, x_test, y_test
