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
import numpy as np
import argparse
from preprocess_for_diagnosis import prepro

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用第三块GPU（从0开始）

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    print("使用GPU " + gpu.name)
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description="Process training parameters.")
parser.add_argument("-batch_size", help="Batch Size for training and testing", type=int, default=64)
parser.add_argument("-num_epoch", help="Num of Epoch for training", type=int, default=50)  # 30就够了
parser.add_argument("-source", help="Source load", type=int, default=0)
parser.add_argument("-target", help="Target load", type=int, default=3)
parser.add_argument("-lr", help="Learning Rate", type=float, default=0.001)
args = parser.parse_args()

source_path = '../{}HP'.format(args.source)
print('source_path', source_path)
x_train, y_train, src_test, y_src_test, x_test, y_test = prepro(d_path=source_path,
                                                                gan_data=None,
                                                                length=1024,
                                                                # number=1000，则训练集中每个样本取500，测试集每个样本取250
                                                                # 算上 imbalance_ratio，正常样本取500，其他故障样本取500/imbalance_ratio
                                                                number=400,  # 10,20,40,100,200,400
                                                                normalization='None',
                                                                rate=[0.5, 0.25, 0.25],
                                                                sampling='random',
                                                                over_sampling='none',
                                                                imbalance_ratio=1,
                                                                )

x_train, src_test, x_test = x_train[:, :, np.newaxis], src_test[:, :, np.newaxis], x_test[:, :, np.newaxis]


def Conv_1D_Block(x, model_width, kernel, strides):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="same")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    return pool


def conv_block(x, num_filters, bottleneck=True):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    if bottleneck:
        num_filters_bottleneck = num_filters * 4
        x = Conv_1D_Block(x, num_filters_bottleneck, 1, 1)

    out = Conv_1D_Block(x, num_filters, 3, 1)

    return out


def dense_block(x, num_filters, num_layers, bottleneck=True):
    for i in range(num_layers):
        cb = conv_block(x, num_filters, bottleneck=bottleneck)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)

    return x


def transition_block(inputs, num_filters):
    x = Conv_1D_Block(inputs, num_filters, 1, 2)
    if x.shape[1] <= 2:
        x = tf.keras.layers.AveragePooling1D(pool_size=1, strides=2, padding="same")(x)
    else:
        x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(x)

    return x


class DenseNet:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False, bottleneck=True):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.bottleneck = bottleneck

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten(name='flatten')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def DenseNet121(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 24, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 16, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model


# Configurations
length = 1024  # Length of each Segment
model_name = 'DenseNet121'  # DenseNet Models
model_width = 16  # Width of the Initial Layer, subsequent layers start from here
num_channel = 1  # Number of Input Channels in the Model
problem_type = 'Classification'  # Classification or Regression
output_nums = 10  # Number of Class for Classification Problems, always '1' for Regression Problems
#
model = DenseNet(length, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling='avg',
                 dropout_rate=False, bottleneck=True) \
    .DenseNet121()

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

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
