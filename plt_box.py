import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 两个模型5次实验结果
five_experiment_result = [[99.99, 99.01, ],
                          [98.23, 97.62, ],
                          [99.6, 95.62, ],
                          [99.73, 94.62, ],
                          [99.3, 98.62, ],
                          ]
df = pd.DataFrame(five_experiment_result, columns=['Model_A', 'Model_B'])
# 箱型图着色
# boxes 箱线
# whiskers 分为数于error bar横线之间的竖线的颜色
# medians 中位线的颜色
# caps error bar 横线的颜色
color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
df.plot.box(grid=True, color=color,  # color 样式填充
            # ylim = [0,1.2],   # y轴刻度范围
            )
df.plot.box(vert=True,  # 是否垂直 默认true
            # positions=[1,4,5,6,8],    # 箱型图占位 相当于箱体之间的间隔
            grid=True,
            color=color,
            )
plt.grid(linestyle='--')
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)
plt.savefig('./pic/box_five_experiment_result.svg', dpi=600)
plt.show()
