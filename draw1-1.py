from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

import matplotlib.pyplot as plt
import numpy
import os

x_data = ['通道1', '通道2', '通道3', '通道4', '通道5', '通道6', ]
names = ['偏移类型A', '偏移类型B', '偏移类型C']

a_f = [118.26084593479361, 7.942321879643713, 0.7169065631867564, 182.3464496326612, 144.52969923104553,
       0.7171030058934309]
a_t = [1.630057125, 8.322631598, 9.945914316, 2.117148789, 2.613360144, 10.66360455]
b_f = [142.62214446189614, 2.0291464652585676, 0.49928201000056216, 94.93793049815059, 148.18899098629328,
       123.71268738733899]
b_t = [1.266246824, 10.97305294, 25.28302315, 4.559479419, 2.856978235, 3.373417164]
c_f = [174.24469722984, 2.142292159176521, 0.598021903875248, 210.73782429151402, 186.07949429962406, 1.923122347964935]
c_t = [1.497955078, 11.93161645, 20.02093721, 3.59921137, 4.149936005, 12.83767229]

bar_width = 0.3
index = np.arange(len(x_data))

# 绘制a
rects1 = plt.bar(index, a_f, bar_width, color='red', label=names[0])
# 绘制b
rects2 = plt.bar(index + bar_width, b_f, bar_width, color='green', label=names[1])
# 绘制c
rects3 = plt.bar(index + bar_width * 2, c_f, bar_width, color='blue', label=names[1])
# X轴标题
plt.xticks(index + bar_width, x_data)

plt.ylim(0,10000)
plt.yscale('symlog')


# 添加数据标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.05, '%.2f' % height, ha='center', va='bottom',fontsize=7)
        # 柱形图边缘用白色填充，纯粹为了美观
        rect.set_edgecolor('white')


add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
# plt.legend(loc='upper left', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5)
plt.legend()

plt.xlabel('通道选择方法')
plt.ylabel('回归测试误差')
plt.title('单个通道上的回归测试误差')

plt.show()
