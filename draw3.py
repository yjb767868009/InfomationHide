from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

import matplotlib.pyplot as plt
import numpy as np
import os

file_path = os.path.join('data3', 'c', 'd3.txt')

data = []
with open(file_path)as file:
    lines = file.readlines()
    for line in lines:
        print(line)
        s = line.strip().split('\t')
        data.append([float(x) for x in s])
data = np.array(data[0])
print(data)

x_data = ['原始数据', '随机选择一个通道', '随机选择两个通道', '随机选择三个通道']
y_data = data

plt.bar(x_data, y_data)

for x, y in zip(x_data, y_data):
    plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')

plt.xlabel('通道选择数量')
plt.ylabel('回归测试误差')
plt.title('偏移类型C在不同通道数量的情况下回归测试误差')

plt.show()
