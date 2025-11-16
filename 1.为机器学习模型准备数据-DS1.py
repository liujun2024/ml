""" 
DS1：仅包含国控点数据和气象数据

"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import _hdf5 as hdf5


os.chdir(sys.path[0])


if __name__ == '__main__':
        
    # 国控点路径
    path_example = r'example.csv'

    # 读取国控点数据
    df_example = pd.read_csv(path_example, encoding='gbk', index_col=0, parse_dates=True)
    print('df_example:\n', df_example)

    # 训练集和测试集划分
    df_train, df_test = train_test_split(df_example, test_size=0.2, shuffle=True, random_state=42)
    print('df_train:\n', df_train)
    print('df_test:\n', df_test)

    # 数据保存路径
    path_h5 = r'h5\DS1.h5'
    if os.path.exists(path_h5):
        os.remove(path_h5)

    # 保存数据
    hdf5.raw2h5(
        df_train=df_train,
        df_test=df_test,
        labels=['O3'],
        path_h5=path_h5,
        dict_attrs={},
        )
