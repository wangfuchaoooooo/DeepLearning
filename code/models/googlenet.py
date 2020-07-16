import keras
from keras.layers import Conv2D, MaxPooling2D,concatenate, Dense, AvgPool2D, Flatten,Dropout,Input
from keras.models import Model

Conv1_kernel_size = 7
Conv1_chs = 64
Conv21_kernel_size = 1
Conv21_chs = 64
Conv2_kernel_size = 3
Conv2_chs = 192
Icp3a_size = (64, 96, 128, 16, 32, 32)
Icp3b_size = (128, 128, 192, 32, 96, 64)
Icp4a_size = (192, 96, 208, 16, 48, 64)
Icp4b_size = (160, 112, 224, 24, 64, 64)
Icp4c_size = (128, 128, 256, 24, 64, 64)
Icp4d_size = (112, 144, 288, 32, 64, 64)
Icp4e_size = (256, 160, 320, 32, 128, 128)
Icp5a_size = (256, 160, 320, 32, 128, 128)
Icp5b_size = (384, 192, 384, 48, 128, 128)
Out_chs1 = 128


class InceptionV1:
    def __init__(self, input_shape=None, classes=None):
        self.input_shape = input_shape
        self.classes = classes

    def InceptionV1_Model(self, input, model_size):
        con11_chs, con31_chs, con3_chs, con51_chs, con5_chs, pool1_chs = model_size

        conv11 = Conv2D(con11_chs, 1, padding='SAME', activation='relu', kernel_initializer='he_normal')(input)

        conv31 = Conv2D(con31_chs, 1, padding='SAME', activation='relu', kernel_initializer='he_normal')(input)
        conv3 = Conv2D(con3_chs, 3, padding='SAME', activation='relu', kernel_initializer='he_normal')(conv31)

        conv51 = Conv2D(con51_chs, 1, padding='SAME', activation='relu', kernel_initializer='he_normal')(input)
        conv5 = Conv2D(con5_chs, 5, padding='SAME', activation='relu', kernel_initializer='he_normal')(conv51)

        pool1 = MaxPooling2D(pool_size=3, strides=1, padding='SAME')(input)
        conv1 = Conv2D(pool1_chs, 1, padding='SAME', activation='relu', kernel_initializer='he_normal')(pool1)

        output = concatenate([conv11, conv3, conv5, conv1], axis=3)
        return output

    def InceptionV1_Out(self, input, name=None):
        pool = AvgPool2D(pool_size=5, strides=3, padding='VALID')(input)
        conv = Conv2D(Out_chs1, 1, padding='SAME', activation='relu', kernel_initializer='he_normal')(pool)

        flat = Flatten()(conv)
        dropout = Dropout(0.3)(flat)
        output = Dense(self.classes, name=name)(dropout)

        return output

    def getNet(self):
        input = Input(shape=self.input_shape, name='input')

        # region conv pool
        conv1 = Conv2D(Conv1_chs, kernel_size=Conv1_kernel_size, padding='SAME', activation='relu', strides=2,
                       kernel_initializer='he_normal')(input)
        pool1 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(conv1)

        conv21 = Conv2D(Conv21_chs, kernel_size=Conv21_kernel_size, padding='SAME', activation='relu',
                        kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(Conv2_chs, kernel_size=Conv2_kernel_size, padding='SAME', activation='relu',
                       kernel_initializer='he_normal')(conv21)
        pool2 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(conv2)
        # endregion

        # region inception3
        inception3a = self.InceptionV1_Model(pool2, Icp3a_size)

        inception3b = self.InceptionV1_Model(inception3a, Icp3b_size)
        pool3 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(inception3b)
        # endregion

        # region inception3
        inception4a = self.InceptionV1_Model(pool3, Icp4a_size)
        output1 = self.InceptionV1_Out(inception4a, 'output1')

        inception4b = self.InceptionV1_Model(inception4a, Icp4b_size)

        inception4c = self.InceptionV1_Model(inception4b, Icp4c_size)

        inception4d = self.InceptionV1_Model(inception4c, Icp4d_size)
        output2 = self.InceptionV1_Out(inception4d, 'output2')

        inception4e = self.InceptionV1_Model(inception4d, Icp4e_size)
        pool4 = MaxPooling2D(pool_size=3, strides=2, padding='SAME')(inception4e)
        # endregion

        # region inception5
        inception5a = self.InceptionV1_Model(pool4, Icp5a_size)

        inception5b = self.InceptionV1_Model(inception5a, Icp5b_size)
        pool5 = MaxPooling2D(pool_size=7, strides=1, padding='SAME')(inception5b)
        # endregion

        # region output
        flat = Flatten()(pool5)
        dropout = Dropout(0.4)(flat)
        output = Dense(self.classes, name='output')(dropout)
        # endregion

        model = Model(inputs=input, outputs=[output, output1, output2])
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'],
                      loss_weights=[0.6, 0.2, 0.2])

        return model
