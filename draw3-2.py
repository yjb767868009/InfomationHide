from pylab import *
import pylab

mpl.rcParams['font.sans-serif'] = ['SimHei']

import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.dpi'] = 300

names = ['偏移类型A', '偏移类型B', '偏移类型C']
x_data = ['原始数据', '固定选择\n一个通道', '固定选择\n一个通道\n随机选择\n一个通道', '固定选择\n一个通道\n随机选择\n两个通道', '固定选择\n一个通道\n随机选择\n三个通道',
          '固定选择\n一个通道\n随机选择\n四个通道', '固定选择\n一个通道\n随机选择\n五个通道', ]
y_data_a = [1.5296833345, 1.630057125, 6.9080764017, 11.3847412641, 15.1973736954, 18.3026696799, 20.9256819994]
y_data_b = [1.0519458345, 1.266246824, 9.5951662893, 16.2249211626, 21.4058978825, 25.1965783957, 28.1176530865]
y_data_c = [1.3366023051, 1.497955078, 10.4344153046, 17.0009447862, 21.8237140937, 25.9079174178, 28.8224985122]
bar_width = 0.3
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
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.05, '%.2f' % height, ha='center', va='bottom',
                 fontsize=7)
        # 柱形图边缘用白色填充，纯粹为了美观
        rect.set_edgecolor('white')


add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
# plt.legend(loc='upper left', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5)
plt.legend()

plt.xlabel('通道选择数量')
# xticks(rotation=45)
plt.ylabel('回归测试误差')
plt.title('使用通道选择算法在不同通道数量的情况下回归测试误差')

plt.show()
