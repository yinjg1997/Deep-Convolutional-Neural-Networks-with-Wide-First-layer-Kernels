import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))  # 不加会导致找不到 utils 包
import argparse
import numpy as np
from preprocess_for_diagnosis import prepro
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
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

model = load_model('./model/WDCNND.h5')
print("=============模型加载完毕============")
model.summary()


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # Blues Accent
    plt.colorbar(fraction=0.045, pad=0.05)

    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 18})
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 18})
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.1f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.xticks(fontproperties='Times New Roman', size=18)


# 显示混淆矩阵
pred = np.argmax(model.predict(x_test), axis=-1)

truelabel = y_test.argmax(axis=-1)  # 将one-hot转化为label
conf_mat = confusion_matrix(y_true=truelabel, y_pred=pred)

# 设置xtick和ytick的方向：in、out、inout
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
plt.figure(figsize=(8, 8))  # 宽 高

plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))

plt.tight_layout()  # 自动调整各子图间距
plt.savefig('./pic/confusion_matrix_wdcnn.svg', dpi=600)
plt.show()
