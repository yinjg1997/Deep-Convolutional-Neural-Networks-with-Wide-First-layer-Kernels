'''
绘图程序
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def draw_data(matfilepath, x, y, z):
    raw_data = scio.loadmat(matfilepath)
    # 读取内容
    signal = ''
    for key, value in raw_data.items():
        if key[5:7] == 'DE':
            signal = value

    time = [i for i in range(2048)]
    axis = np.random.randint(2048)

    # Plot colors numbers
    ax = plt.subplot(x, y, z)
    ax.plot(time, signal[axis:axis + 2048], color='mediumblue')

    plt.ylabel('A(mm)', fontdict={'family': 'Times New Roman', 'size': 18}, )
    plt.xlabel('Sampling points', fontdict={'family': 'Times New Roman', 'size': 18}, )
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.xticks(fontproperties='Times New Roman', size=18)


# 设置xtick和ytick的方向：in、out、inout
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(12, 12))  # 宽12，高8

draw_data('./1HP/12k_Drive_End_B007_1.mat', x=5, y=2, z=1)
draw_data('./1HP/12k_Drive_End_B014_1.mat', x=5, y=2, z=2)
draw_data('./1HP/12k_Drive_End_B021_1.mat', x=5, y=2, z=3)
draw_data('./1HP/12k_Drive_End_IR007_1.mat', x=5, y=2, z=4)
draw_data('./1HP/12k_Drive_End_IR014_1.mat', x=5, y=2, z=5)
draw_data('./1HP/12k_Drive_End_IR021_1.mat', x=5, y=2, z=6)
draw_data('./1HP/12k_Drive_End_OR007@6_1.mat', x=5, y=2, z=7)
draw_data('./1HP/12k_Drive_End_OR014@6_1.mat', x=5, y=2, z=8)
draw_data('./1HP/12k_Drive_End_OR021@6_1.mat', x=5, y=2, z=9)
draw_data('./1HP/normal_1.mat', x=5, y=2, z=10)
plt.tight_layout()  # 自动调整各子图间距
plt.savefig('./pic/1hp_ten_faults_view.svg', dpi=600)
plt.show()
