import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))  # 不加会导致找不到 utils 包
import tensorflow as tf

import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # 导入sklearn库的RandomForestClassifier函数
from sklearn.metrics import classification_report
from sklearn import metrics
from preprocess_for_diagnosis import prepro
from sklearn import model_selection
from sklearn.metrics import confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 使用第三块GPU（从0开始）

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    print("使用GPU " + gpu.name)
    tf.config.experimental.set_memory_growth(gpu, True)
import argparse

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

print("random_forest...")

clf = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_split=5)  # 恒定,无fft
# clf = RandomForestClassifier(n_estimators=300,max_depth=16,min_samples_split=5) # # 变工况fft
# clf = RandomForestClassifier(n_jobs=8,n_estimators=100) # 变工况

# print('The accuracy of RandomForest:')
# print(np.mean(score))

clf.fit(x_train, y_train)
# score = model_selection.cross_val_score(clf, src_test, y_src_test, cv=5, scoring="accuracy")

y_pred = clf.predict(x_test)
y_pred_pro = clf.predict_proba(x_test)
# print(classification_report(y_true, y_pred))
accuracy = metrics.accuracy_score(y_test, y_pred)
# accuracy = classification_report(y_true, y_pred)

print("accuracy_score:", accuracy)

# # 绘制混淆矩阵
# def plot_confusion_matrix(cm, classes,
#                           title='混淆矩阵',
#                           cmap=plt.cm.Greens):
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     # font = {'family': 'STSong', 'size': 12}
#     # plt.title(title, fontdict=font)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.yticks(fontproperties='Times New Roman', size=18)
#     plt.xticks(fontproperties='Times New Roman', size=18)
#     # plt.ylabel('True label')
#     # plt.xlabel('Predicted label')
#     # plt.ylabel('真实类别', fontdict=font)
#     # plt.xlabel('预测类别', fontdict=font)
#     plt.savefig('../images/RF/2confusion_matrix_' + str(args.source) + str(args.target) + '.svg', dpi=300)
#     plt.show()
#
#
# # 显示混淆矩阵
# truelabel = y_test  # 将one-hot转化为label  ndarray (1000,)
# conf_mat = confusion_matrix(y_true=truelabel, y_pred=y_pred)
#
# # 设置xtick和ytick的方向：in、out、inout
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# plt.figure()
# plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))
