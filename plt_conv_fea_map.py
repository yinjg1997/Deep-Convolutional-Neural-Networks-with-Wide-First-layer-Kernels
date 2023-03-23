# coding: utf-8
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from preprocess_for_diagnosis import prepro
import argparse

parser = argparse.ArgumentParser(description="Process training parameters.")
parser.add_argument("-operation", help="load", type=int, default=0)
args = parser.parse_args()

data_path = './{}HP'.format(args.operation)
print('data_path', data_path)
x_train, y_train, src_test, y_src_test, x_test, y_test = prepro(d_path=data_path,
                                                                gan_data=None,
                                                                length=2048,
                                                                # number=1000，则训练集中每个样本取500，测试集每个样本取250
                                                                # 算上 imbalance_ratio，正常样本取500，其他故障样本取500/imbalance_ratio
                                                                number=400,  # 20,30,40,100,200,400
                                                                normalization='None',
                                                                rate=[0.5, 0.25, 0.25],
                                                                sampling='random',
                                                                over_sampling='none',
                                                                imbalance_ratio=1,
                                                                )
print(x_test.shape)

x_test = x_test
x_test = x_test[:, :, np.newaxis]


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch, num):
    feature_map = img_batch
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[1]
    print(num_pic)
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, i]
        print('feature_map_split: ', feature_map_split.shape)
        feature_map_combination.append(feature_map)



    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    axis('off')  # 关闭坐标轴
    plt.savefig("./pic/img_prefix_{}feature_map_sum.png".format(num))
    plt.show()

if __name__ == "__main__":

    base_model = load_model('./model/WDCNND.h5')
    base_model.summary()



    for index in range(0, 5):
        if index == 0:
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('max_pooling1d').output)
        else:
            model = Model(inputs=base_model.input, outputs=base_model.get_layer('max_pooling1d_{}'.format(index)).output)
        print(x_test[0:1].shape)
        conv_features = model.predict(x_test[0:1])
        print(conv_features.shape)

        feature = conv_features.reshape(conv_features.shape[1:])
        print('index: ', index)
        visualize_feature_map(feature, index, )

