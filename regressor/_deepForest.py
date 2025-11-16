from __future__ import annotations
from typing import List
"""
需要python<=3.9
numpy==1.23.1

pip install numpy==1.23.1 deep-forest
"""

import numpy as np
import pandas as pd
from zipfile import ZIP_LZMA
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import metrics
# 导入深度森林
from deepforest import CascadeForestClassifier, CascadeForestRegressor
from sklearn.ensemble import RandomForestRegressor
import skops.io as sio
from _hdf5 import lc2hdf
from _utils import predict


class DeepForest():
    """ 深度森林回归 """

    def __init__(self, x_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray, cpu=4):

        # 初始化参数
        self.x_train = x_train
        self.y_train = y_train
        self.cpu = cpu

        # 学习曲线初始参数
        self.dict_params_init = {
                    'n_estimators': [1, 2, 4, 8],
                    'n_trees':[50, 80, 150, 200, 500],
                    'max_layers': [2, 5, 10, 20, 50, 100],
                    'predictor': ['forest', 'xgboost', 'lightgbm'],
                    'n_bins': [2, 10, 50, 100, 200, 255],
                    'bin_type': ['percentile', 'interval'],
                    'max_depth': [1, 2, 5, 10, 20, 50, 100],
                    'min_samples_split': np.arange(start=2, stop=21, step=3),
                    'min_samples_leaf': np.arange(start=1, stop=20, step=3),
                    'n_tolerant_rounds': [1, 2, 5]
            }

        # 初始参数-通常可以得到较的模型表现
        self.dict_params_overfitting = {
                    'n_estimators': 2,
                    'n_trees': 100,
                    'max_layers': 20,
                    'predictor': 'forest',
                    'n_bins': 255,
                    'bin_type': 'percentile',
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'n_tolerant_rounds': 2,
        }

        # 用于储存最优参数，key为list_p中的元素，value为对应的最优值
        self.dict_params_best = dict()

        # 用于储存训练过程的所有结果，key为list_p中的元素，value为对应的pd.Series（index为参数值，data为对应的模型表现）
        self.dict_params_all = dict()

        # 保存训练好的模型
        self.model = None

        # 对训练集进行预测的r2、rmse
        self.r2_train = None
        self.rmse_train = None

        # 对验证集进行预测的r2、rmse
        self.r2_test = None
        self.rmse_test = None

        # 预测器predictor参数
        self.dict_predictor_xgboost = {
                'objective': 'reg:squarederror', 
                'n_estimators': 100, 
                'learning_rate': 0.1, 
                'max_depth': 10,
                'sample_weight': None,
            }
        
        self.dict_predictor_lightgbm = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9
            }

        # 当前模型训练的超参数
        self.dict_params_j = None

    def fit(self):
        """ 依次对各个参数进行优化 """

        list_p = list(self.dict_params_init.keys())

        for i, p in enumerate(list_p):

            # 参数
            todo_p = self.dict_params_init[p]
            if not isinstance(todo_p, (list, np.ndarray)):
                # 参数储存至字典
                self.dict_params_best[p] = todo_p
                continue

            """ 逐个参数进行训练 """
            list_score = []
            for j in todo_p:

                # 准备参数
                self.dict_params_j = self.dict_params_best.copy()
                self.dict_params_j.update({p: j})
                self.dict_params_j.update({k: self.dict_params_overfitting[k] for k in list_p if k not in self.dict_params_j.keys()})
                
                # print(f'调参:{p} | {self.dict_params_j}', end=' ')
                print(f'调参({i+1}/{len(list_p)}): {p} | {self.dict_params_j}', end=' ')

                # 训练
                score_j = self.tune_manual()
                
                # 训练得分保存至列表, deepforest暂时无score, 用训练集预测r2代替
                list_score.append(score_j)

                print('Score: %.3f' % score_j)

            # 训练结果保存至pd.Series
            series_score = pd.Series(data=list_score, index=todo_p, name=p)

            # 参数储存至字典
            self.dict_params_best[p] = series_score.idxmax()
            self.dict_params_all[p] = series_score

        """ 生成最优模型 """
        # 判断predictor
        if self.dict_params_best['predictor'] == 'xgboost':
            params = self.dict_predictor_xgboost
        elif self.dict_params_best['predictor'] == 'lightgbm':
            params = self.dict_predictor_lightgbm
        elif self.dict_params_best['predictor'] == 'forest':
            params = {}
        else:
            raise KeyError('predictor参数错误:%s' % self.dict_params_best['predictor'])
        
        # 创建模型
        self.model = create_model(
            dict_param=self.dict_params_best, 
            predictor_kwargs=params, 
            cpu=4,
            )
        
        # 拟合
        self.model.fit(X=self.x_train, y=self.y_train)

    def tune_manual(self):
        """ 训练某一组参数 """
        
        # 判断predictor
        if self.dict_params_j['predictor'] == 'xgboost':
            params = self.dict_predictor_xgboost
        elif self.dict_params_j['predictor'] == 'lightgbm':
            params = self.dict_predictor_lightgbm
        elif self.dict_params_j['predictor'] == 'forest':
            params = {}
        else:
            raise KeyError('predictor:%s' % self.dict_params_j['predictor'])
        
        # 创建模型
        model = create_model(dict_param=self.dict_params_j, cpu=self.cpu, predictor_kwargs=params)

        # 训练模型
        model.fit(self.x_train, self.y_train)

        # 预测, 使用r2作为评估指标
        r2_train = predict(model=model, x=self.x_train, y=self.y_train)['r2']

        return r2_train

    def save_model(self, path_skops):
        """ 保存模型 """

        print('模型保存...', end=' ')

        # 保存
        sio.dump(obj=self.model, file=path_skops, compression=ZIP_LZMA, compresslevel=3)

        print('完成！')

    def save_params(self, path_hdf5):
        """ 保存调参数据至hdf5文件 """

        print('学习曲线保存...', end=' ')
        lc2hdf(path_hdf5=path_hdf5, dict_all_params=self.dict_params_all, dict_best_param=self.dict_params_best)
        print('完成！')

    def predict_train(self):
        """ 预测训练集 """

        dict_predict = predict(model=self.model, x=self.x_train, y=self.y_train)

        self.r2_train, self.rmse_train, self.y_predict_train = map(dict_predict.get, ['r2', 'rmse', 'predict'])

    def predict_test(self, x: np.ndarray, y: np.ndarray):
        """ 预测测试集 """

        dict_predict = predict(model=self.model, x=x, y=y)

        self.r2_test, self.rmse_test, self.y_predict_test = map(dict_predict.get, ['r2', 'rmse', 'predict'])


def create_model(dict_param: dict, predictor_kwargs: dict, cpu=1):
    """ 根据参数创建模型

        dict_param: 给定的参数，如：
                {'n_estimators': 1000,
                 'max_depth': 100,
                 'min_samples_split': 2,
                 'min_samples_leaf': 1,
                 'max_features': 'sqrt',
                 'max_samples': 1.0,
                 }

        x_train: pd.DataFrame，待训练的数据，自变量
        y_train: pd.Series，待训练的数据，因变量
        cpu: int，调用的CPU核心数量，默认为：1

        return: model

    """

    # 建立随机森林回归模型
    model = CascadeForestRegressor(
        n_bins=dict_param['n_bins'],  # 每个特征分箱的数量
        bin_subsample=200000,
        bin_type=dict_param['bin_type'],
        max_layers=dict_param['max_layers'],
        criterion='mse',
        n_estimators=dict_param['n_estimators'],    # 随机森林回归树的数量
        n_trees=dict_param['n_trees'],
        max_depth=dict_param['max_depth'],
        min_samples_split=dict_param['min_samples_split'],
        min_samples_leaf=dict_param['min_samples_leaf'],
        use_predictor=True,
        predictor=dict_param['predictor'],
        predictor_kwargs=predictor_kwargs,
        backend='custom',
        n_tolerant_rounds=dict_param['n_tolerant_rounds'],
        delta=1E-5,
        partial_mode=False,
        n_jobs=cpu,
        random_state=42,
        verbose=0,
    )

    return model


# def predict(model, x: np.ndarray, y: np.ndarray):
#     """ 使用随机森林模型进行预测

#         path_skops: os.PathLike，模型文件的具体路径
#         data: pd.DataFrame，待预测的数据，含有索引，最后一列为因变量，其它列为自变量

#         return: {
#             'r2': r2,
#             'rmse': rmse,
#             'predict': np.1darray,
#         }

#     2023-06-19 v1
#     2024-07-19 v2
#     单进程单线程
#     """

#     # 预测
#     y_predict = model.predict(X=x)

#     # 计算root mean squared error均方根误差
#     rmse = metrics.root_mean_squared_error(y_true=y, y_pred=y_predict)

#     # computes the coefficient of determination, usually denoted as R2
#     r2 = metrics.r2_score(y_true=y, y_pred=y_predict)

#     dict_result = {
#         'rmse': rmse,
#         'r2': r2,
#         'predict': y_predict,
#     }

#     # 返回数据
#     return dict_result


# def demo(x_train, x_test, y_train, y_test):

#     """ 默认参数回归 """

#     # 模型初始化
#     model = CascadeForestRegressor(random_state=1, n_jobs=8)

#     base_estimator = [RandomForestRegressor()]
#     model.set_estimator(estimators=base_estimator, n_splits=5)

#     # 训练模型
#     model.fit(x_train, y_train)

#     # 预测
#     y_pred_train = model.predict(x_train)
#     y_pred_test = model.predict(x_test)

#     # 计算RMSE
#     rmse_train = metrics.root_mean_squared_error(y_true=y_train, y_pred=y_pred_train)
#     rmse_test = metrics.root_mean_squared_error(y_true=y_test, y_pred=y_pred_test)

#     # 计算相关性R2
#     r2_train = metrics.r2_score(y_true=y_train, y_pred=y_pred_train)
#     r2_test = metrics.r2_score(y_true=y_test, y_pred=y_pred_test)

#     # 返回数据
#     return {
#         "rmse_train": rmse_train,
#         "rmse_test": rmse_test,
#         "r2_train": r2_train,
#         "r2_test": r2_test,
#         'pred_train': y_pred_train,
#         'pred_test': y_pred_test,
#     }


if __name__ == "__main__":

    """ deomo测试 """
    X, y = fetch_california_housing(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    result_df = demo(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(result_df)
