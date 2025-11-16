""" 高斯回归，极慢，性能差，直接删除 """

from __future__ import annotations
import os, sys
import math
import numpy as np
import pandas as pd
from zipfile import ZIP_LZMA

from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
# import sklearn.model_selection as optimizers
from sklearn.model_selection import validation_curve, cross_val_score, learning_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class Gauss():
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.result = GaussR(x_train, x_test, y_train, y_test)
        

def GaussR(x_train, x_test, y_train, y_test):

    """ 默认参数回归 """

    # 标准化数据
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 定义核函数
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))

    # 模型初始化
    model = GaussianProcessRegressor(kernel=kernel)

    # 训练模型
    model.fit(x_train, y_train)

    # 预测
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # 计算RMSE
    # rmse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train, squared=False)
    # rmse_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test, squared=False)
    rmse_train = root_mean_squared_error(y_true=y_train, y_pred=y_pred_train)
    rmse_test = root_mean_squared_error(y_true=y_test, y_pred=y_pred_test)

    # 计算相关性R2
    r2_train = r2_score(y_true=y_train, y_pred=y_pred_train)
    r2_test = r2_score(y_true=y_test, y_pred=y_pred_test)

    # 返回数据
    return {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
        'pred_train': y_pred_train,
        'pred_test': y_pred_test,
    }


if __name__ == '__main__':

    pass
