import keras
from keras import backend as K


def LRN(alpha=1e-4, k=2, beta=0.75, n=5):
    def f(X):
        b, r, c, ch = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(square, ((0, 0), (half, half)), data_format='channels_first')
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, :, :, i:i + int(ch)]
        scale = scale ** beta
        return X / scale

    return keras.layers.Lambda(f, output_shape=lambda input_shape: input_shape)

def AlexNet(input_shape=None, classes=None):
    inp = keras.layers.Input(input_shape)

    x = keras.layers.Conv2D(96, kernel_size=(11, 11), strides=4, activation='relu')(inp)
    x = LRN()(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = keras.layers.Conv2D(256, kernel_size=(5, 5), strides=1, activation='relu', padding='same')(x)
    x = LRN()(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = keras.layers.Conv2D(384, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(384, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    output = keras.layers.Dense(classes, activation='softmax')(x)

    model = keras.models.Model(inputs=inp, outputs=output)

    return model


