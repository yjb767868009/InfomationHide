# coding=gbk
# 把x归一化，注意，最后一项分类标签不要归一化
import pandas as pd
import pickle
import math
import os
from pandas import read_csv

filenameDataSet = os.path.join('data1', '1.csv')
df = pd.read_csv(filenameDataSet, header=None)

for iii in range(df.shape[0]):  # 把训练集中的数据按照总的数据归一化
    for jjj in range(2, df.shape[1]):
        tty = df.loc[iii, jjj]
        df.iloc[iii, jjj] = math.log(tty, 1024)

# y=(x-MinValue)/(MaxValue-MinValue)
dump_path = os.path.join('data1', '1.pkl')
pickle.dump(df, open(dump_path, 'wb'))
# print(df)
