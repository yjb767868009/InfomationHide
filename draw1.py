from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

import matplotlib.pyplot as plt
import numpy
import os


def draw():
    # names = ['原始数据', '选择通道算法', '随机选择通道']
    # x_data = ['偏移类型A', '偏移类型B', '偏移类型C']
    # y_data_a = [1.5296833345237142, 1.0519458345, 1.3366023051]
    # y_data_b = [1.630057125, 1.2662468241, 1.4979550779]
    # y_data_c = [5.8821194201, 8.052032956, 9.0062213996]

    x_data = ['原始数据', '选择通道算法', '随机选择通道']
    names= ['偏移类型A', '偏移类型B', '偏移类型C']
    y_data_a = [1.5296833345237142, 1.630057125, 5.8821194201]
    y_data_b = [1.0519458345, 1.2662468241, 8.052032956]
    y_data_c = [1.3366023051, 1.4979550779, 9.0062213996]

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

    plt.xlabel('通道选择方法')
    plt.ylabel('回归测试误差')
    plt.title('单个通道上的回归测试误差')

    plt.show()


def draw_a():
    x_data = ['原始数据', '选择通道算法', '随机选择通道']
    y_data = [1.5296833345237142, 1.630057125, 5.8821194201]

    plt.bar(x_data, y_data)

    for x, y in zip(x_data, y_data):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    plt.xlabel('通道选择方法')
    plt.ylabel('回归测试误差')
    plt.title('偏移类型A在单个通道上的回归测试误差')

    plt.show()


def draw_b():
    x_data = ['原始数据', '选择通道算法', '随机选择通道']
    y_data = [1.0519458345, 1.2662468241, 8.052032956]

    plt.bar(x_data, y_data)

    for x, y in zip(x_data, y_data):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    plt.xlabel('通道选择方法')
    plt.ylabel('回归测试误差')
    plt.title('偏移类型B在单个通道上的回归测试误差')

    plt.show()


def draw_c():
    x_data = ['原始数据', '选择通道算法', '随机选择通道']
    y_data = [1.3366023051, 1.4979550779, 9.0062213996]

    plt.bar(x_data, y_data)

    for x, y in zip(x_data, y_data):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    plt.xlabel('通道选择方法')
    plt.ylabel('回归测试误差')
    plt.title('偏移类型C在单个通道上的回归测试误差')

    plt.show()


if __name__ == '__main__':
    draw()
