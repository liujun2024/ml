""" 支持向量机(Support Vector Machine, SVM)相关的类和函数 """

from __future__ import annotations
import os, sys
import math
import numpy as np
import pandas as pd
from zipfile import ZIP_LZMA

from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp

from sklearn.preprocessing import StandardScaler
# import sklearn.model_selection as optimizers
from sklearn.model_selection import validation_curve, cross_val_score, learning_curve, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import skops.io as sio
import _hdf5 as h5
import _utils as utils


mp.set_start_method('spawn', force=True)
torch.cuda.empty_cache()


class NeuralNetwork(BaseEstimator, RegressorMixin):

    def __init__(self, x_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray, cv=5, cpu=4):

        # 初始化参数
        # self.x_train = torch.tensor(x_train, dtype=torch.float32)
        # self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.x_train, self.y_train = x_train, y_train
        # self.dataset = TensorDataset(self.x_train, self.y_train)
        # self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.cv = cv
        self.cpu = cpu

        # 特征数量
        self.input_size = x_train.shape[1]
        
        # 需要拟合的参数
        # self.list_params_for_fitting = ['n_estimators', 'max_bin']

        # 对需要拟合的参数，寻找最优值时，取最大值的百分数
        # self.percent_max = 0.99

        # 学习曲线初始参数
        self.dict_params_init = {
            'hidden_size': [5, 10, 20, 30],
            'epochs': [50, 100, 200, 500],
            'batch_size': [10, 20],
            'lr': [0.001, 0.01, 0.1],
            # 'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 20],
        }

        # # 初始参数-通常可以得到较的模型表现
        # self.dict_params_overfitting = {
        #     'n_estimators': 100,
        #     'max_depth': 6,
        #     'max_leaves': 0,
        #     'max_bin': 256,
        #     # 'grow_policy': 'depthwise',
        #     # 'learning_rate': 0.1,
        #     # 'booster': 'gbtree',
        # }

        # 用于储存最优参数，key为list_p中的元素，value为对应的最优值
        self.dict_params_best = dict()

        # 用于储存训练过程的所有结果，key为list_p中的元素，value为对应的pd.Series（index为参数值，data为对应的模型表现）
        self.dict_params_all = dict()

        # 用于储存交叉验证n_estimators和max_depth的拟合曲线
        self.dict_fitting = dict()

        # 保存训练好的模型
        self.model = None

        # 对训练集进行预测的r2、rmse
        self.r2_train = None
        self.rmse_train = None

        # 对验证集进行预测的r2、rmse
        self.r2_test = None
        self.rmse_test = None

        # 当前模型训练的超参数
        self.dict_params_j = None

        # 当前正在训练的超参数
        self.current_p = None

    def fit(self):
        """ 网格搜索调参 """
        model = PyTorchRegressor(input_size=self.input_size)

        random_search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=self.dict_params_init,
            n_iter=10,
            n_jobs=self.cpu,
            cv=self.cv, 
            verbose=10,
            random_state=42,
            )
        
        random_search.fit(self.x_train, self.y_train)

        # 输出最佳参数和得分
        print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
        
        # 使用最佳参数进行预测
        best_model = random_search.best_estimator_
        predictions = best_model.predict(self.x_train)

        # 计算r2和rmse
        

        """ 依次对各个参数进行优化 """

        list_p = list(self.dict_params_init.keys())

        for i, self.current_p in enumerate(list_p):

            # 参数
            todo_p = self.dict_params_init[self.current_p]
            if not isinstance(todo_p, (list, np.ndarray)):
                # 参数储存至字典
                self.dict_params_best[self.current_p] = todo_p
                continue
                
            """ 学习曲线/交叉验证 """
            # 准备用于创建模型的参数
            self.dict_params_j = self.dict_params_best.copy()
            self.dict_params_j.update({self.current_p: todo_p[0]})
            self.dict_params_j.update({k: self.dict_params_overfitting[k] for k in list_p if k not in self.dict_params_j.keys()})

            # 准备用于print的参数
            self.dict_params_j2 = self.dict_params_j.copy()
            self.dict_params_j2[self.current_p] = todo_p

            print(f'调参({i+1}/{len(list_p)}): {self.current_p} | {self.dict_params_j2}')
            
            # 学习曲线
            score = self.tune_curve()

            # 交叉验证得分存入pd.DataFrame
            df_score = pd.DataFrame(data=score, index=todo_p)

            # 训练参数存入字典
            self.dict_params_all[self.current_p] = df_score

            # 最优参数存入字典
            if self.current_p in self.list_params_for_fitting:
                # 如果是n_estimators和max_depth，则拟合得到最优值
                self.dict_params_best[self.current_p] = self.fit_logistic()
            else:
                self.dict_params_best[self.current_p] = df_score['test_mean'].idxmax()

        """ 生成最优模型 """
        # 创建模型
        self.model = self.create_model(dict_param=self.dict_params_best)
        
        # 拟合
        self.model.fit(X=self.x_train, y=self.y_train)

    def tune_curve(self):
        """ 获取学习曲线 """

        # 创建模型
        model = self.create_model(dict_param=self.dict_params_j)

        # 使用validation_curve获取学习曲线数据
        score_train, score_test = validation_curve(
            estimator=model,
            X=self.x_train, 
            y=self.y_train,
            cv=self.cv,
            n_jobs=self.cpu,
            param_name=self.current_p, 
            param_range=self.dict_params_init[self.current_p],
        )

        # 计算平均值和标准差
        dict_score = {
            'train_mean': np.mean(score_train, axis=1),
            'train_std': np.std(score_train, axis=1),
            'test_mean': np.mean(score_test, axis=1),
            'test_std': np.std(score_test, axis=1),
        }

        return dict_score

    def create_model(self, dict_param: dict):
        """ 根据参数创建模型 """

        model = xgb.XGBRegressor(
            n_estimators=100 if 'n_estimators' not in dict_param.keys() else dict_param['n_estimators'],
            max_depth=6 if 'max_depth' not in dict_param.keys() else dict_param['max_depth'],
            max_leaves=0 if 'max_leaves' not in dict_param.keys() else dict_param['max_leaves'],
            tree_method='hist',
            max_bin=256 if 'max_bin' not in dict_param.keys() else dict_param['max_bin'],
            grow_policy='depthwise' if 'grow_policy' not in dict_param.keys() else dict_param['grow_policy'],
            learning_rate=0.1 if 'learning_rate' not in dict_param.keys() else dict_param['learning_rate'],
            booster='gbtree' if 'booster' not in dict_param.keys() else dict_param['booster'],
        )

        return model

    def fit_logistic(self):
        """ 逻辑回归拟合 """
        
        # 拟合
        params, df = utils.fitting_logistic(self.dict_params_all[self.current_p].iloc[:, [0, 2]])
        
        # 取train_mean的最大值
        y_max_score = df.max(axis=0).iloc[1] * self.percent_max

        # 获得对应的x值
        x_max_score = utils.function_logistic_inverse(y_max_score, *params.iloc[:, 1].to_numpy())

        # 拟合曲线存入字典
        self.dict_fitting[self.current_p] = df
        
        # 返回最优x和y
        return math.ceil(x_max_score)

    def save_model(self, path_skops):
        """ 保存模型 """

        print('模型保存...', end=' ')

        # 保存
        sio.dump(obj=self.model, file=path_skops, compression=ZIP_LZMA, compresslevel=3)

        print('完成！')

    def save_params(self, path_hdf5):
        """ 保存调参数据至hdf5文件 """

        print('学习曲线保存...', end=' ')
        h5.lc2hdf(path_hdf5=path_hdf5, dict_all_params=self.dict_params_all, dict_best_param=self.dict_params_best)
        print('完成！')

    def predict_train(self):
        """ 预测训练集 """

        dict_predict = utils.predict(model=self.model, x=self.x_train, y=self.y_train)

        self.r2_train, self.rmse_train, self.y_predict_train = map(dict_predict.get, ['r2', 'rmse', 'predict'])

    def predict_test(self, x: np.ndarray, y: np.ndarray):
        """ 预测测试集 """

        dict_predict = utils.predict(model=self.model, x=x, y=y)

        self.r2_test, self.rmse_test, self.y_predict_test = map(dict_predict.get, ['r2', 'rmse', 'predict'])


class PyTorchRegressor(BaseEstimator, RegressorMixin):
    """  """
    def __init__(self, input_size: int, hidden_size=10, epochs=100, batch_size=10, lr=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(input_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
        return predictions.flatten()


# 自定义神经网络模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# def regressor(x_train, x_test, y_train, y_test):

#     """ 默认参数回归 """

#     # 标准化数据
#     scaler = StandardScaler()
#     x_train = scaler.fit_transform(x_train)
#     x_test = scaler.transform(x_test)

#     # 模型初始化
#     model = MLPRegressor(verbose=True, learning_rate='adaptive')

#     # 训练模型
#     model.fit(x_train, y_train)

#     # 预测
#     y_pred_train = model.predict(x_train)
#     y_pred_test = model.predict(x_test)

#     # 计算RMSE
#     # rmse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train, squared=False)
#     # rmse_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test, squared=False)
#     rmse_train = root_mean_squared_error(y_true=y_train, y_pred=y_pred_train)
#     rmse_test = root_mean_squared_error(y_true=y_test, y_pred=y_pred_test)

#     # 计算相关性R2
#     r2_train = r2_score(y_true=y_train, y_pred=y_pred_train)
#     r2_test = r2_score(y_true=y_test, y_pred=y_pred_test)

#     # 返回数据
#     return {
#         "rmse_train": rmse_train,
#         "rmse_test": rmse_test,
#         "r2_train": r2_train,
#         "r2_test": r2_test,
#         'pred_train': y_pred_train,
#         'pred_test': y_pred_test,
#     }


if __name__ == '__main__':

    pass
