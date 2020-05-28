from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

import matplotlib.pyplot as plt
import numpy as np
import os

names = ['偏移类型A', '偏移类型B', '偏移类型C']
x_data = ['原始数据', '随机选择一个通道', '随机选择两个通道', '随机选择三个通道',
          '随机选择四个通道', '随机选择五个通道', '随机选择六个通道']
y_data_a = [1.5296833345, 5.8821194201, 9.9311251705, 13.2545567211, 16.1583804219, 18.6963720674, 20.9256819994]
y_data_b = [1.0519458345, 8.052032956, 13.9300598136, 18.5221370008, 22.2674455518, 25.873486595, 28.1176530865]
y_data_c = [1.3366023051, 9.0062213996, 14.9402358476, 19.458342884, 23.1854645188, 26.0981614693, 28.8224985122]
bar_width = 0.2
index = np.arange(len(x_data))

# 绘制a
rects1 = plt.bar(index, y_data_a, bar_width, color='red', label=names[0])
# 绘制b
rects2 = plt.bar(index + bar_width, y_data_b, bar_width, color='green', label=names[1])
# 绘制c
rects3 = plt.bar(index + bar_width * 2, y_data_c, bar_width, color='blue', label=names[1])
# X轴标题
plt.xticks(index + bar_width, x_data)


# 添加数据标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.05, '%.2f' % height, ha='center', va='bottom')
        # 柱形图边缘用白色填充，纯粹为了美观
        rect.set_edgecolor('white')


add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
# plt.legend(loc='upper left', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5)
plt.legend()

plt.xlabel('通道选择数量')
plt.ylabel('回归测试误差')
plt.title('在不同通道数量的情况下回归测试误差')

plt.show()
