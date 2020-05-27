# experiment 算测试集的平均误差
#### 前提是：现在已经有一个标准数据集了，大概像这样 时间，106.4150858224653,75,22,24,101,393,229的很多行，现在要load相应的model去算准确率
from collections import Counter  # 取三次输入的最大值
import pandas as pd
import math
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from pandas import DataFrame
import os

aa = [2]
outputNodeName = 'outputtttttttt_node'
namestr = ["1108", "1109", "1110"]
filerrr = 'data2/c.csv'  # 数据集的存放位置
# 加载模型

import tensorflow as tf

tf.reset_default_graph()
graph = tf.get_default_graph()
sess = tf.Session()
modelpath = 'model/'  # 模型存放位置
saver = tf.train.import_meta_graph(modelpath + 'model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint(modelpath))
print(modelpath)

if os.path.exists(filerrr):  # 归一化
    df = pd.read_csv(filerrr, header=None)  # 融合用的角度
    # X_Data=dg.iloc[:,0:6]
    for iii in range(df.shape[0]):  # 把训练集中的数据按照总的数据归一化
        for jjj in range(2, 8):
            tty = df.loc[iii, jjj]
            df.iloc[iii, jjj] = math.log(tty, 1024)

    X_Data = df.iloc[:, 2:8]
    Y_Data = df.iloc[:, 1]
    print("XData:", X_Data)
    print("Y_Data", Y_Data)
    X_Data2 = X_Data.as_matrix()
    X_Data3 = X_Data2.tolist()

    Y_Data2 = Y_Data.as_matrix()
    print(Y_Data2)
    rrrYTR = Y_Data2.shape[0]

    Y_Data3 = Y_Data2.reshape(rrrYTR, 1)
    Y_Data4 = Y_Data3.tolist()
    X = None  # input
    yhat = None  # output
    X = graph.get_tensor_by_name('input_plllllllllllaceholder_x:0')
    Y = graph.get_tensor_by_name('ouput_plllllllllllaceholder_y:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    ouputNode = str(outputNodeName) + ':0'
    yhat = graph.get_tensor_by_name(ouputNode)
    aac = sess.run(yhat, feed_dict={X: X_Data3, keep_prob: 1.0})
    df = pd.DataFrame(aac)

    sum = 0
    pp = 0
    for i in range(df.shape[0]):
        jjji = df.iloc[i, 0]
        # print("jjji:",jjji)
        yyyi = Y_Data4[i]
        # print("yyyi",yyyi)
        error = abs(yyyi[0] - jjji)
        sum += error
    print(sum / df.shape[0])
