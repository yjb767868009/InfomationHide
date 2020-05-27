import pandas as pd
import math
import datetime as dt
import os
import pickle
from pandas import read_csv

lujing = 'data1/'
batch_size = 400

# filenameDataSet='/home/wdenxw/Train2/TFNN/0128/DealFCN0128FF1.csv'
# df = pd.read_csv(filenameDataSet, header=None)


import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import pickle
import os
import pandas as pd
import numpy as np
import datetime
import random
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import cmath
import math
import numpy

XTrain_path = lujing + 'XTraincache.pkl'
YTrain_path = lujing + 'YTraincache.pkl'
XTest_path = lujing + 'XTestcache.pkl'
YTest_path = lujing + 'YTestcache.pkl'

dump1_path = lujing + 'XTraincache.pkl'
dump2_path = lujing + 'XTestcache.pkl'
dump3_path = lujing + 'YTraincache.pkl'
dump4_path = lujing + 'YTestcache.pkl'

if os.path.exists(dump1_path):
    try:
        X_train = pickle.load(open(dump1_path, 'rb'))
        print(X_train[0])

    except EOFError:
        print("EOF")
if os.path.exists(dump2_path):
    try:
        X_test = pickle.load(open(dump2_path, 'rb'))
        print("type Xtest:", type(X_test))
    except EOFError:
        print("EOF")
if os.path.exists(dump3_path):
    try:
        Y_train = pickle.load(open(dump3_path, 'rb'))
    except EOFError:
        print("EOF")

if os.path.exists(dump4_path):
    try:
        Y_test = pickle.load(open(dump4_path, 'rb'))
    except EOFError:
        print("EOF")

dump_path = lujing + '1.pkl'
if (not (os.path.exists(dump1_path))):
    dg = pickle.load(open(dump_path, 'rb'))
    X_train, X_test, Y_train, Y_test = train_test_split(dg.iloc[:, 2:8], dg.iloc[:, 1], train_size=0.8)

# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# y = x.pow(2) + 0.2 * torch.rand(x.size())

X_test2 = X_test.as_matrix()
X_test3 = X_test2.tolist()

Y_test2 = Y_test.as_matrix().transpose()
rrrYTE = Y_test2.shape[0]
Y_test3 = Y_test2.reshape(rrrYTE, 1)
Y_test4 = Y_test3.tolist()

Y_testn = np.mat(Y_test4)
X_testn = np.mat(X_test2)

n_batch = len(X_train) // batch_size

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

# x_pred = [[120,5,85,120,5,85]]

Hidden_n = 512
tf_x = tf.placeholder(tf.float32, [None, 6], name="input_plllllllllllaceholder_x")  # input x
tf_y = tf.placeholder(tf.float32, [None, 1], name="ouput_plllllllllllaceholder_y")  # input y
keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # drop out node

# neural network layers

# W1 = tf.Variable(np.random.randn(7,Hidden_n)) / np.sqrt(7/2)

l1 = tf.layers.dense(tf_x, Hidden_n, tf.nn.relu, name="inputtttttttt_node")  # hidden layer
# L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
l1_drop = tf.nn.dropout(l1, keep_prob)
# W2 = tf.Variable(np.random.randn(Hidden_n,Hidden_n)) / np.sqrt(Hidden_n/2)
l2 = tf.layers.dense(l1_drop, Hidden_n, tf.nn.relu)  # hidden layer
l2_drop = tf.nn.dropout(l2, keep_prob)
# W3 = tf.Variable(np.random.randn(Hidden_n,Hidden_n)) / np.sqrt(Hidden_n/2)

# W7 = tf.Variable(np.random.randn(Hidden_n,Hidden_n)) / np.sqrt(Hidden_n/2)
l7 = tf.layers.dense(l2_drop, Hidden_n, tf.nn.relu)  # hidden layer
l7_drop = tf.nn.dropout(l7, keep_prob)
# W8 = tf.Variable(np.random.randn(Hidden_n,1)) / np.sqrt(Hidden_n/2)
l8 = tf.layers.dense(l7_drop, 1)
# output = tf.layers.dense(l7, 1, name="outputtttttttt_node")                     # output layer
Uui = tf.add(l8, l8)
output = tf.subtract(Uui, l8, name="outputtttttttt_node")
loss = tf.losses.mean_squared_error(tf_y, output)  # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)
train_op = optimizer.minimize(loss)
saveryy = tf.train.Saver()  # control training and others
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # initialize var in graph
# tf.reset_default_graph()
print("xinhao")


def seq_train(X_train, y_train, batch_size, batch):
    #  buchang=len(X_train)//batch_size
    number = []
    for i in range(batch_size):
        number.append(i + batch * batch_size)
    aa = np.array(number)

    # rnd_indices = np.random.randint(0, len(X_train), batch_size)
    # print("lenrnd_indices",rnd_indices)
    X_batch = []
    y_batch = []
    Xtrain = X_train.values
    Ytrain = Y_train.values
    for i in aa:
        X_batch.append(Xtrain[i])
        y_batch.append([Ytrain[i]])
    # y_batch = y_train[rnd_indices]
    return X_batch, y_batch


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
# train_op = optimizer.minimize(loss)
loss_list = []
bias_list = []
acc_list = []

step = 0
while (True):
    for batch in range(n_batch):
        batch_xs, batch_ys = seq_train(X_train, Y_train, batch_size, batch)
        # print("batchys:",batch_ys)
        # print("ytest:",Y_test4)
        # batch_ys=Y_train.next_batch(batch_size)
        # train and net output
        _, l, pred = sess.run([train_op, loss, output], {tf_x: batch_xs, tf_y: batch_ys, keep_prob: 1.0})
        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Input/x", "Input/y", "Output/predictions"])
    if (step % 500 == 0):
        now_time = dt.datetime.now().strftime('%F %T')
        print('Now Time is:' + now_time)
        print('loss is: ' + str(l))
        pred1 = sess.run(output, {tf_x: X_test3, keep_prob: 1.0})
        # pred1 =  sess.run(output, feed_dict={tf_x: X_test3})  # (-1, 3)

        # print('prediction is:' + str(pred))
        k = 0
        k1 = 0
        sum1 = 0
        for i in range(len(Y_test4)):
            jjji = pred1[i]
            yyyi = Y_test4[i]
            sum1 += abs(yyyi[0] - jjji[0])
            if (abs(yyyi[0] - jjji[0]) < 5):
                # print(Reverse(yyyi[0]),Reverse(jjji[0]))
                k1 += 1
                # break
            # if abs(yyyi-jjji) < 1:

        print(sum1 / len(Y_test4))
        print(k1 / len(Y_test4))

        loss_list.append(l)
        bias_list.append(sum1 / len(Y_test4))
        acc_list.append(k1 / len(Y_test4))

        lujingSave = lujing + 'model.ckpt'
        saveryy.save(sess, lujingSave)
        print("successSave!")
    step += 1

with open(lujing + 'loss.txt', 'w') as f:
    f.write()
