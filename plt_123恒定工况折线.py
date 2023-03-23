import matplotlib.pyplot as plt
import numpy as np
import argparse

test_acc1 = np.load('./acc/10月06日BP_acc.npy') * 100
test_acc2 = np.load('./acc/10月06日GRU_acc.npy') * 100
test_acc3 = np.load('./acc/10月06日CNN_1D_acc.npy') * 100
test_acc4 = np.load('./acc/10月06日MSCNN_acc.npy') * 100
test_acc5 = np.load('./acc/10月06日CA_MSCNN_GRU_acc.npy') * 100
test_acc6 = np.load('./acc/10月06日CA_MSCNN_acc.npy') * 100

plt.figure(figsize=(8, 6))
# 设置最大值和最小值(可以两个都设置，也可以只设置一个，只设置一个的时候要显式声明)
plt.xlim(0, 10.5)
plt.ylim(20, 105)

x1 = np.arange(0, len(test_acc1))
plt.plot(x1 + 1, test_acc1, '-rp')
plt.plot(x1 + 1, test_acc2, '-cs')
plt.plot(x1 + 1, test_acc3, '-g^')
plt.plot(x1 + 1, test_acc4, '-b*')
plt.plot(x1 + 1, test_acc6, '-m+')
plt.plot(x1 + 1, test_acc5, '-yd')

plt.yticks(fontproperties='Times New Roman', size=18)
plt.xticks(fontproperties='Times New Roman', size=18)
plt.ylabel('Accuracy(%)', fontproperties='Times New Roman', size=18)
plt.xlabel('Epochs', fontproperties='Times New Roman', size=18)

label = ['BP',
         'BiGRU',
         '1DCNN',
         'MSCNN',
         'CAMSCNN',
         'CAMSCNN-BiGRU',
         ]
plt.legend(labels=label, loc='best',
           prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 18}, )
plt.savefig('./pic/epoch_point_.svg', dpi=600)
plt.show()
