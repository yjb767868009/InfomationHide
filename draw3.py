import matplotlib.pyplot as plt
import numpy as np
import os

file_path = os.path.join('data3', 'a', 'd3.txt')

data = []
with open(file_path)as file:
    lines = file.readlines()
    for line in lines:
        print(line)
        s = line.strip().split('\t')
        data.append([float(x) for x in s])
data = np.array(data[0])
print(data)

x_data = ['原始数据', '选择通道算法', '随机选择通道']
y_data = data

plt.bar(x_data, y_data)
plt.xlabel('')
plt.ylabel('回归测试误差')
plt.title('')

plt.show()