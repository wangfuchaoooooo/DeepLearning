import numpy as np


def read_data(file_path):
    data = np.load(file_path)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    return x_train, y_train, x_test, y_test


# x_train, y_train, x_test, y_test = read_data('./data/mnist/mnist.npz')
# print(x_train.shape)
