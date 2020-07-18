import numpy as np # linear algebra
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input,Activation, ZeroPadding2D, MaxPooling2D, Flatten, merge

# 下面是自定义的图像标准化代码。
# 比循环要快
# 常数项是吴恩达在他的 UFLDL 教程中建议的
def per_image_normalization(X, constant=10.0, copy=True):
    if copy:
        X_res = X.copy()
    else:
        X_res = X

    means = np.mean(X, axis=1)
    variances = np.var(X, axis=1) + constant
    X_res = (X_res.T - means).T
    X_res = (X_res.T / np.sqrt(variances)).T
    return X_res

###########################################################################################
# 现在简单介绍残差网络。
##Salient features of the Resnet implemented
# 1.As originally proposed, no MaxPooling Layers are used.
#   Down Sampling is done by varying the strides and kernels of the Convolutional2D layers only
# 2.Even though the network is so deep, the number of parameters is very small (as you'll notice later)
# 3.We use only 2 residual blocks.
# 4.The original proposal didn't use Dropout.
#   The implemented model has Dropout between the final Dense Layers.
# 5.The layers are well labelled and it makes it a little easier to see what's actually going on
# 6.NO HYPERPARAMETERS WERE TUNED, PERIOD. Every parameter,
#   including the number of Feature maps at every Convolution, was arbitrarily chosen.
#   Therefore, there is a lot of space for hyperparam Tuning.
# 7.The implementation has been done using the Keras Functional API.
###############################################################################
# lets get to it and define the function that will make up the network

def ResNet(input_shape=None, classes=None):
    # In order to make things less confusing, all layers have been declared first, and then used

    # declaration of layers
    input_img = Input(input_shape=input_shape, name='input_layer')
    zeroPad1 = ZeroPadding2D((1, 1), name='zeroPad1', dim_ordering='th')
    zeroPad1_2 = ZeroPadding2D((1, 1), name='zeroPad1_2', dim_ordering='th')
    layer1 = Convolution2D(6, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv', dim_ordering='th')
    layer1_2 = Convolution2D(16, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv2', dim_ordering='th')
    zeroPad2 = ZeroPadding2D((1, 1), name='zeroPad2', dim_ordering='th')
    zeroPad2_2 = ZeroPadding2D((1, 1), name='zeroPad2_2', dim_ordering='th')
    layer2 = Convolution2D(6, 3, 3, subsample=(1, 1), init='he_uniform', name='l1_conv', dim_ordering='th')
    layer2_2 = Convolution2D(16, 3, 3, subsample=(1, 1), init='he_uniform', name='l1_conv2', dim_ordering='th')

    zeroPad3 = ZeroPadding2D((1, 1), name='zeroPad3', dim_ordering='th')
    zeroPad3_2 = ZeroPadding2D((1, 1), name='zeroPad3_2', dim_ordering='th')
    layer3 = Convolution2D(6, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv', dim_ordering='th')
    layer3_2 = Convolution2D(16, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv2', dim_ordering='th')

    layer4 = Dense(64, activation='relu', init='he_uniform', name='dense1')
    layer5 = Dense(16, activation='relu', init='he_uniform', name='dense2')

    final = Dense(classes, activation='softmax', init='he_uniform', name='classifier')

    # declaration completed

    first = zeroPad1(input_img)
    second = layer1(first)
    second = BatchNormalization(0, axis=1, name='major_bn')(second)
    second = Activation('relu', name='major_act')(second)

    third = zeroPad2(second)
    third = layer2(third)
    third = BatchNormalization(0, axis=1, name='l1_bn')(third)
    third = Activation('relu', name='l1_act')(third)

    third = zeroPad3(third)
    third = layer3(third)
    third = BatchNormalization(0, axis=1, name='l1_bn2')(third)
    third = Activation('relu', name='l1_act2')(third)

    res = merge([third, second], mode='sum', name='res')

    first2 = zeroPad1_2(res)
    second2 = layer1_2(first2)
    second2 = BatchNormalization(0, axis=1, name='major_bn2')(second2)
    second2 = Activation('relu', name='major_act2')(second2)

    third2 = zeroPad2_2(second2)
    third2 = layer2_2(third2)
    third2 = BatchNormalization(0, axis=1, name='l2_bn')(third2)
    third2 = Activation('relu', name='l2_act')(third2)

    third2 = zeroPad3_2(third2)
    third2 = layer3_2(third2)
    third2 = BatchNormalization(0, axis=1, name='l2_bn2')(third2)
    third2 = Activation('relu', name='l2_act2')(third2)

    res2 = merge([third2, second2], mode='sum', name='res2')

    res2 = Flatten()(res2)

    res2 = layer4(res2)
    res2 = Dropout(0.4, name='dropout1')(res2)
    res2 = layer5(res2)
    res2 = Dropout(0.4, name='dropout2')(res2)
    res2 = final(res2)
    model = Model(input=input_img, output=res2)

    return model