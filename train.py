# %%
import numpy as np
import tensorflow as tf
from noise import wgn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from preprocess_for_diagnosis import prepro
from model import Train_N_samples
import argparse

parser = argparse.ArgumentParser(description="Process training parameters.")
"""
改变工况
"""
parser.add_argument("-cross", help="cross diagnosis", type=bool, default=False)
parser.add_argument("-source", help="load", type=int, default=0)
parser.add_argument("-target", help="load", type=int, default=3)
"""
改变样本量
"""
parser.add_argument("-sample", help="s", type=int, default=1000)  #  sample代表每种样本的数量, 这里就代表总共有4000s,80%训练
"""
改变噪声
"""
parser.add_argument("-dB", help="noise dB", type=str, default="null")  # null -4 -2 0 2 4 6 8
args = parser.parse_args()

source_dir = './{}HP'.format(args.source)
train_data, train_label, valid_data, valid_label, test_data, test_label = prepro(d_path=source_dir,
                                                                                 gan_data=None,
                                                                                 length=2048,
                                                                                 number=args.sample,  # 400
                                                                                 normalization='None',
                                                                                 # 大最小值归一化'minmax', 均值归一化'mean', 归一化为0-1之间'0-1'
                                                                                 rate=[0.8, 0.1, 0.1],
                                                                                 sampling='random',
                                                                                 over_sampling='none',
                                                                                 imbalance_ratio=1,
                                                                                 )
# 进行变工况实验
if args.cross:
    target_dir = './{}HP'.format(args.target)
    _, _, valid_data, valid_label, test_data, test_label = prepro(d_path=target_dir,
                                                                  gan_data=None,
                                                                  length=2048,
                                                                  number=args.sample,  # 400
                                                                  normalization='None',
                                                                  rate=[0.8, 0.1, 0.1],
                                                                  sampling='random',
                                                                  over_sampling='none',
                                                                  imbalance_ratio=1,
                                                                  )

# 进行噪声实验
if args.dB != "null":
    print("added noise~")
    data = np.vstack((valid_data, test_data))
    print("noise data shape{}", format(data.shape))
    data = wgn(data, int(args.dB))  # -4 -2 0 2 4 6 8
    valid_data = data[0:len(data) // 2]
    test_data = data[len(data) // 2:]

train_data, valid_data, test_data = train_data[:, :, np.newaxis], valid_data[:, :, np.newaxis], test_data[:, :,
                                                                                                np.newaxis]
print('train_data.shape:{},\
      \nvalid_data.shape:{},\
      \ntest_data.shape:{}'.format(train_data.shape, valid_data.shape, test_data.shape))

# train D
signal_len = 2048
batch_size = 64
epochs = 100
output_dir = './model'
name = 'D'

train_samples = Train_N_samples(output_dir,
                                train_data, train_label, valid_data, valid_label, test_data, test_label,
                                signal_len, batch_size, name, epochs=epochs)

model = train_samples.WDCNN_model()
# model_2 = train_samples.CNN_1D_model() #
# model_3 = train_samples.BP_model() #
# model_4 = train_samples.GRU_model() #
# model_5 = train_samples.LSTM_model() #
# model_6 = train_samples.WDCNN_AdaBN_model() #
