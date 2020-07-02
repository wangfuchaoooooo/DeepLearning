import keras


def LeNet(input_shape=None, classes=None):
    inp = keras.layers.Input(input_shape)  # 输入层
    conv1 = keras.layers.Conv2D(filters=6, kernel_size=(5, 5),
                                strides=1, padding='valid',
                                activation='sigmoid')(inp)  # 第一层卷积
    pooling1 = keras.layers.AvgPool2D(pool_size=(2, 2),
                                      strides=2,
                                      padding='valid')(conv1)  # 第一层池化
    conv2 = keras.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                strides=1, padding='valid',
                                activation='sigmoid')(pooling1)  # 第二层卷积
    pooling2 = keras.layers.AvgPool2D(pool_size=(2, 2),
                                      strides=2,
                                      padding='valid')(conv2)  # 第二层池化
    flatten = keras.layers.Flatten()(pooling2)  # 扁平化
    fc1 = keras.layers.Dense(120, activation='sigmoid')(flatten)  # 第一个全连接层
    fc2 = keras.layers.Dense(84, activation='sigmoid')(fc1)  # 第二个全连接层
    output = keras.layers.Dense(classes, activation='softmax')(fc2)  # 第三个全连接层（输出层）

    model = keras.models.Model(inp, output)
    return model
