import matplotlib.pyplot as plt
import numpy as np

p1 = plt.figure(figsize=(12, 8), dpi=600)  # 画布大小，分辨率；
# plt.rcParams['font.sans-serif'] = 'SimHei  '  # 仿宋
# 设置最大值和最小值(可以两个都设置，也可以只设置一个，只设置一个的时候要显式声明)
plt.ylim(50, 105)

###设置坐标轴的粗细
ax = plt.gca()  # 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(0)  ###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(0)  ###设置右边坐标轴的粗细
# 修改刻度线线粗细width参数，修改刻度字体labelsize参数
plt.tick_params(width=2)

plt.xlabel('Task', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 22}, )
plt.ylabel('Accuracy(%)', fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 22}, )

x = ['1-2', '1-3', '2-1', '2-3', '3-1', '3-2', 'Avg']  # 7个横坐标

x1 = np.array([i for i in range(0, 42, 6)])  # 每个横坐标有5个模型，多出一个用于分割不同横坐标空隙，也就是6*7=42
# 将每四个柱状图之间空一格
x2 = x1 + 1
x3 = x1 + 2
x4 = x1 + 3
x5 = x1 + 4
x6 = x1 + 5

# 6 个模型在 7 个横坐标上的诊断结果
y1 = [75.3, 84.6, 87, 87, 73.6, 76.7, 80.7]  # BiGRU
y2 = [90, 82.3, 89, 81.3, 86.6, 89.6, 86.4]  # 1DCNN
y3 = [87, 86.7, 93, 86, 78.6, 86.3, 86.2]  # MSCNN
y4 = [98.6, 99.3, 98, 99.7, 89, 92.6, 96.3]  # CAMSCNN
y5 = [98.3, 99.3, 98, 99.9, 97.3, 96, 98.1]  # CAMSCNN-BiGRU

# colors = ['black','tomato','yellow','cyan','blue', 'lime', 'r', 'violet','m','peru','olivedrab','hotpink']#设置散点颜色
plt.bar(x1, y1, label="BiGRU", edgecolor='black', hatch='-')
plt.bar(x2, y2, label="1DCNN", edgecolor='black', hatch='+')
plt.bar(x3, y3, label="MSCNN", edgecolor='black', hatch='x')
plt.bar(x4, y4, label="CAMSCNN", edgecolor='black', hatch='o')
plt.bar(x5, y5, label="CAMSCNN-BiGRU", edgecolor='black', hatch='*')
plt.bar(x6, 0)

Label_Com = ["BiGRU", "1DCNN", "MSCNN", "CAMSCNN", "CAMSCNN-BiGRU"]  ##图例名称
plt.legend(labels=Label_Com, loc='upper center', labelspacing=0.4,
           columnspacing=0.4, markerscale=2, ncol=12,
           bbox_to_anchor=(.5, 1.1),
           # borderaxespad = 0.,
           prop={'family': ['Times New Roman'], 'weight': 'bold', 'size': 18},
           handletextpad=0.1, edgecolor='black')

plt.xticks(x1 + 2, x, fontproperties='Times New Roman', weight='bold', size=22)  # +2.5是让下标在四个柱子中间
plt.yticks(fontproperties='Times New Roman', size=22)
plt.savefig('./pic/cross_accuracy.svg', dpi=600)
plt.show()
