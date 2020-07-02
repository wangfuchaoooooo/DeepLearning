import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as f:
        dict_ = pickle.load(f, encoding='latin1')
    return dict_


def read_data(path):
    x_train = []
    y_train = []
    for i in range(1, 6):
        file = path+'/data_batch_' + str(i)
        data = unpickle(file)
        label = data['labels']
        image = data['data']
        image = image.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")  # (C, H, W) ---> (H, W, C)
        x_train.append(image)
        y_train.append(label)
    x_train = np.array(x_train).reshape(-1, 32, 32, 3)
    y_train = np.array(y_train).reshape(-1)

    test_file = path+'/test_batch'  # 测试数据集
    test_data = unpickle(test_file)
    x_test = np.array(test_data['data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    y_test = np.array(test_data['labels'])
    return x_train, y_train, x_test, y_test



