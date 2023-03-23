#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  2019.9.11 16:32:19 
@author: Zkj
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))  # 不加会导致找不到 utils 包
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.disable_eager_execution()
from tensorflow import keras
import numpy as np
from preprocess_for_diagnosis import prepro

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 使用第三块GPU（从0开始）

parser = argparse.ArgumentParser(description="Process training parameters.")
parser.add_argument("-batch_size", help="Batch Size for training and testing", type=int, default=64)
parser.add_argument("-num_epoch", help="Num of Epoch for training", type=int, default=50)  #
parser.add_argument("-source", help="Source load", type=int, default=0)
parser.add_argument("-target", help="Target load", type=int, default=0)
parser.add_argument("-sample", help="s", type=int, default=400)  # 20,30,40,100,200,400
parser.add_argument("-dB", help="noise dB", type=int, default=4)  # -4 -2 0 2 4 6 8

args = parser.parse_args()
source_path = '../{}HP'.format(args.source)
print('source_path', source_path)
x_train, y_train, src_test, y_src_test, x_test, y_test = prepro(d_path=source_path,
                                                                gan_data=None,
                                                                length=1024,
                                                                # number=1000，则训练集中每个样本取500，测试集每个样本取250
                                                                # 算上 imbalance_ratio，正常样本取500，其他故障样本取500/imbalance_ratio
                                                                number=args.sample,  # 20,30,40,100,200,400
                                                                normalization='None',
                                                                rate=[0.5, 0.25, 0.25],
                                                                sampling='random',
                                                                over_sampling='none',
                                                                imbalance_ratio=1,
                                                                )

x_train, src_test, x_test = x_train[:, :, np.newaxis], src_test[:, :, np.newaxis], x_test[:, :, np.newaxis]

np.random.seed(813306)


def build_resnet(input_shape, n_feature_maps, nb_classes):
    print('build conv_x')
    x = keras.layers.Input(shape=(input_shape))
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv1D(n_feature_maps, 8, 1, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps, 1, 1, padding='same')(x)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps * 2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps * 2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, 1, 1, padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps * 2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps * 2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, 1, 1, padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    full = keras.layers.GlobalAveragePooling1D()(y)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    print('        -- model was built.')
    return x, out


# nb_classes = len(np.unique(y_test))
# batch_size = min(x_train.shape[0]/10, 16)

# x_train = x_train.reshape(x_train.shape + (1,))
# x_test = x_test.reshape(x_test.shape + (1,))

print(x_train.shape[1:])
x, y = build_resnet(x_train.shape[1:], 16, 10)
model = keras.models.Model(inputs=x, outputs=y)
optimizer = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.001)
hist = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.num_epoch,
                 verbose=1, validation_data=(src_test, y_src_test),
                 callbacks=[reduce_lr])
# model.save("../saved_models/ResNet/resnet_trained_model_"+ str(args.source) + str(args.target) +".h5")
# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])

# with open("../log/ResNetlog", "a") as f:
#     f.write(str(args.source) + "--->" + str(args.target) + "|0%" + ": " + str(score[1]) + "\n")
