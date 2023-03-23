import os

# dir_now=sys.path[0]
# dir_up=os.path.dirname(dir_now)
# dir_upup=os.path.dirname(dir_up)
# print(dir_now)
# print(dir_up)
# print(dir_upup)
import sys

sys.path.append(os.path.dirname(sys.path[0]))  # 不加会导致找不到 utils 包
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import argparse
import numpy as np
from preprocess_for_diagnosis import prepro

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第三块GPU（从0开始）

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    print("使用GPU " + gpu.name)
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description="Process training parameters.")
parser.add_argument("-batch_size", help="Batch Size for training and testing", type=int, default=64)
parser.add_argument("-num_epoch", help="Num of Epoch for training", type=int, default=50)  # 30就够了
parser.add_argument("-source", help="Source load", type=int, default=0)
parser.add_argument("-target", help="Target load", type=int, default=0)
parser.add_argument("-sample", help="s", type=int, default=30)  # 20,30,40,100,200,400
parser.add_argument("-lr", help="Learning Rate", type=float, default=0.001)
parser.add_argument("-dB", help="noise dB", type=int, default=-4)  # -4 -2 0 2 4 6 8
args = parser.parse_args()

source_path = '../{}HP'.format(args.source)
print('source_path', source_path)
x_train, y_train, src_test, y_src_test, x_test, y_test = prepro(d_path=source_path,
                                                                gan_data=None,
                                                                length=2048,
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


def Conv_1D_Block(x, model_width, kernel):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, padding='same', kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


class VGG:
    def __init__(self, length=1024, num_channel=1, num_filters=16, problem_type='Classification', output_nums=10,
                 dropout_rate=False):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.dropout_rate = dropout_rate

    def VGG11(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        # Block 1
        x = Conv_1D_Block(inputs, self.num_filters * (2 ** 0), 64)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 2
        x = Conv_1D_Block(x, self.num_filters * (2 ** 1), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 3
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 2), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 4
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Block 5
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        x = Conv_1D_Block(x, self.num_filters * (2 ** 3), 3)
        if x.shape[1] <= 2:
            x = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="valid")(x)
        else:
            x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)

        # Fully Connected (MLP) block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        # Create model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model


# Configurations
length = 2048  # Length of each Segment
model_name = 'VGG11'  # DenseNet Models
model_width = 16  # Width of the Initial Layer, subsequent layers start from here
num_channel = 1  # Number of Input Channels in the Model
problem_type = 'Classification'  # Classification or Regression
output_nums = 10  # Number of Class for Classification Problems, always '1' for Regression Problems
#
model = VGG(length, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, dropout_rate=False) \
    .VGG11()

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
              loss='categorical_crossentropy', metrics=['accuracy'])

# 开始模型训练
hist = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.num_epoch,
                 verbose=1, validation_data=(src_test, y_src_test))

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])
# model.save("../saved_models/CNN/cnn_trained_model_"+ str(args.source) + str(args.target) +".h5")
# with open("../log/CNNlog", "a") as f:
#     f.write(str(args.source) + "--->" + str(args.target) + "|0%" + ": " + str(score[1]) + "\n")
