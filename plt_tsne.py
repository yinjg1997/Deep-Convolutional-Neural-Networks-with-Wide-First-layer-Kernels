import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))  # 不加会导致找不到 utils 包
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from preprocess_for_diagnosis import prepro
import argparse

parser = argparse.ArgumentParser(description="Process training parameters.")
parser.add_argument("-sample", help="s", type=int, default=400)  # 20,30,40,100,200,400
args = parser.parse_args()

data_dir = './0HP'
x_train, y_train, src_test, y_src_test, x_test, y_test = prepro(d_path=data_dir,
                                                                gan_data=None,
                                                                length=2048,
                                                                number=args.sample,  # 400
                                                                normalization='None',
                                                                rate=[0.5, 0.25, 0.25],
                                                                sampling='random',
                                                                over_sampling='none',
                                                                imbalance_ratio=1,
                                                                )
x_train, src_test, x_test = x_train[:, :, np.newaxis], src_test[:, :, np.newaxis], x_test[:, :, np.newaxis]

x_test = x_test[0:600, :, :]
y_test = y_test[0:600, :]

model = load_model('./model/WDCNND.h5')  # dense_1
model_name = "WDCNN"

model.summary()  # 可以查看每一层的名字

# 查看中间结果，必须要先声明个函数式模型
'''
WDCNN
'input_1'
'conv1d'
'conv1d_1'
'conv1d_2'
'conv1d_3'
'conv1d_4'
'dense_1'
'''
layer_name = 'dense_1' # 输出wdcnn不同层的tsne
dense_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


def plot_embedding_2d(X, y):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(5, 5))

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.hot(y[i] / 10.), # hot Accent
                 fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 15})
    # plt.axis('off')
    plt.yticks(fontproperties='Times New Roman', size=15)
    plt.xticks(fontproperties='Times New Roman', size=15)


print("Computing t-SNE embedding")
features = dense_layer_model(x_test)
# print("features.shape: " +str(features.shape[0]))
if (layer_name != 'dense_1' and model_name == 'WDCNN'):
    features = features.numpy()
    features = features.reshape(features.shape[0], len(features[1]) * len(features[1][0]))

pred = np.argmax(y_test, axis=-1)
# t-SNE对原始图像的降维与可视化
tsne = TSNE(n_components=2, init='pca', random_state=1).fit_transform(features)

plot_embedding_2d(tsne[:, 0:2], pred, )
plt.savefig('./pic/{}.svg'.format(model_name), dpi=600)
plt.show()
