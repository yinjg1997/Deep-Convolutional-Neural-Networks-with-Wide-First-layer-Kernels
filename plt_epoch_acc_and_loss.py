import matplotlib.pyplot as plt
import numpy as np

# 同样可改为loss
train_acc  = np.load('./acc/WDCNN_train_acc.npy')
val_acc = np.load('./acc/WDCNN_val_acc.npy')


p1 = plt.figure(figsize=(8,8))
plt.plot(train_acc)
plt.plot(val_acc)


plt.xticks(fontproperties='Times New Roman', weight='bold', size=18)
plt.yticks(fontproperties='Times New Roman', weight='bold', size=18)
plt.ylabel('Accuracy', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 22})
plt.xlabel('Epoch', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 22})
plt.tight_layout()   # 自动调整各子图间距
label = ['Train_acc',
         'Valid_acc']
plt.legend(labels = label, loc='lower right',
           prop={'family': 'Times New Roman','weight': 'bold','size': 22},)
plt.savefig('./pic/epoch_acc.svg', dpi=600)
plt.show()