# coding=gbk
# ��x��һ����ע�⣬���һ������ǩ��Ҫ��һ��
import pandas as pd
import pickle
import math
import os
from pandas import read_csv

filenameDataSet = os.path.join('data1', '1.csv')
df = pd.read_csv(filenameDataSet, header=None)

for iii in range(df.shape[0]):  # ��ѵ�����е����ݰ����ܵ����ݹ�һ��
    for jjj in range(2, df.shape[1]):
        tty = df.loc[iii, jjj]
        df.iloc[iii, jjj] = math.log(tty, 1024)

# y=(x-MinValue)/(MaxValue-MinValue)
dump_path = os.path.join('data1', '1.pkl')
pickle.dump(df, open(dump_path, 'wb'))
# print(df)
