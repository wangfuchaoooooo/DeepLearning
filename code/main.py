import os
import datetime
import time
import numpy as np
import keras
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings
from utils import load_data, get_flops, create_directory, \
    save_hist, save_metrics, save_confusion
from models.lenet import LeNet

# 添加GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
warnings.filterwarnings("ignore")

print('[信息打印] 数据加载......')
x_train, y_train, x_test, y_test = load_data('mnist')
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
class_ = np.unique(y_train)
nb_class = len(class_)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)

print('[信息打印] 创建模型......')
model = LeNet(input_shape=x_train.shape[1:], classes=nb_class)
model.summary()
print('[信息打印]模型FLOPs..... ', get_flops())

# 构建模型保存文件夹
time_ = str(datetime.datetime.now()).split('.')[0]. \
    replace(':', '_').replace(' ', '_').replace('-', '_')
# 创建文件夹
model_result_dir = create_directory(os.path.join('./checkpoint', time_))
# 构建模型名
checkpoint_name = os.path.join(model_result_dir, 'model-io-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}-'
                                                 'val_acc{val_acc:.5f}.h5f')
# 设置模型保存参数
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_name, monitor='val_acc',
                                                   save_best_only=True, verbose=1, mode='max')
# TensorBoard构建
TensorBoard = keras.callbacks.TensorBoard(log_dir=os.path.join(model_result_dir, 'log'))
# 梯度平滑
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
callbacks = [reduce_lr, model_checkpoint, TensorBoard]

print('[信息打印] 编译模型......')
# sgd = SGD(lr=0.0001, momentum=0.9)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')
print('[信息打印] 训练模型......')
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=100,
                    validation_data=(x_valid, y_valid),
                    callbacks=callbacks)
save_hist(model_result_dir, history)

print('[信息打印] 保存模型结构和参数')
model_json = model.to_json()
# 保存模型（不含权重）
with open(os.path.join(model_result_dir, 'model_architecture.json'), 'w') as fp:
    fp.write(model_json)
# 保存权重
model.save_weights(os.path.join(model_result_dir, 'model_weights.h5'), overwrite=True)
# '模型评估及预测'
print('[信息打印] 评估模型...')
score = model.evaluate(x_test, y_test, batch_size=32)
print('test score: {}'.format(score[0]))
print('test accuracy:{}'.format(score[1]))

print('[信息打印] 模型预测...')
start_time = time.time()
y_pre = model.predict(x_test)
duration = time.time() - start_time
y_pre = np.argmax(y_pre, axis=1)
y_test = np.argmax(y_test, axis=1)
df_metrics = save_metrics(model_result_dir, y_pre, y_test, duration)
save_confusion(y_test, y_pre, class_, save_path=model_result_dir)
print('[信息打印] 预测信息...\n', df_metrics)
