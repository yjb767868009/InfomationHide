from pylab import *
import pylab

mpl.rcParams['font.sans-serif'] = ['SimHei']

import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.dpi'] = 300


def read_file(x, y):
    data = []
    file_path = os.path.join('data3', x, y)
    with open(file_path)as file:
        lines = file.readlines()
        for line in lines:
            s = line.strip().split(' ')
            data.append(float(s[1]))
    return np.array(data)


data = []
for i in range(1, 7):
    data.append(read_file('a', str(i) + '.txt'))
    data.append(read_file('b', str(i) + '.txt'))
    data.append(read_file('c', str(i) + '.txt'))

plt.boxplot(data, showfliers=False)

plt.xlabel('通道选择数量')
# xticks(rotation=45)
plt.ylabel('回归测试误差')
plt.title('在不同通道数量的情况下回归测试误差')

plt.show()
