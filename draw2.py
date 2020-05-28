from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

import matplotlib.pyplot as plt
import numpy as np
import os

file_path = os.path.join('data3', 'a', 'd2.txt')

data = []
with open(file_path)as file:
    lines = file.readlines()
    for line in lines:
        s = line.strip().split('\t')
        data.append([float(x) for x in s])
data = np.array(data)
print(data)

fig, ax = plt.subplots()
im = ax.imshow(data)
ax.set_xticks(np.arange(6))
ax.set_yticks(np.arange(6))
ax.set_xticklabels([i for i in range(1, 7)])
ax.set_yticklabels([i for i in range(1, 7)])

for i in range(6):
    for j in range(6):
        text = ax.text(j, i, round(data[i, j], 2), ha="center", va="center", color="w")

ax.set_title("偏移类型A在两个通道组合后的回归测试误差")
fig.tight_layout()
plt.show()
