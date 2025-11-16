""" 决策树算法（DecisionTree）"""

from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
import zipfile
import warnings
from typing import Literal
from pathlib import Path

import skops.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from auto_shap.auto_shap import generate_shap_values
# from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from ._superclass import ShapBasedExplainer

from .. import _hdf5 as h5
from .. import _utils as utils
from .. import _plot as plot
# import _hdf5 as h5
# import _utils as utils
# import _plot as plot


# 模型关键字缩写
suffix_kw = 'dt'


class DecisionTree:
    """ 决策树 """

    def __init__(self, path_h5: os.PathLike, cv=5, cpu=4):

        # 警告
        warnings.warn('此类将在后续版本中删除，请使用DT类代替！', DeprecationWarning)

        # hdf5文件路径
        self.path_h5 = path_h5

        # 初始化目录
        self.__init_dir()

        # 交叉验证cv-fold
        self.cv = cv

        # 并行处理的cpu核心数
        self.cpu = cpu

        # 需要拟合的参数
        self.list_params_for_fitting = []

        # 对需要拟合的参数，寻找最优值时，取最大值的百分数
        self.percent_max = 0.98

        # 学习曲线初始参数
        self.dict_params_init = {
            'criterion': ['absolute_error', 'friedman_mse', 'poisson', 'squared_error'],
            'splitter': ['best', 'random'],
            'max_depth': [1, 2, 3, 4, 5, 7, 9, 12, 15, 20],
            'min_samples_split': [2, 5, 10, 20, 50],
            'min_samples_leaf': [1, 2, 5, 10, 20],
            'max_features': [0.2, 0.4, 0.6, 0.8, 1.0],
            'max_leaf_nodes': [1, 5, 10, 20],
        }

        # 初始参数-通常可以得到较的模型表现
        self.dict_params_overfitting = {
            'criterion': 'squared_error',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'max_leaf_nodes': None,
        }

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

        # 读取hdf5文件训练集和测试集数据
        self.h5rw = h5.HDF5RW(path_h5=self.path_h5)

        # 赋值：训练集x
        self.x_train = self.h5rw.x_train
        
        # 赋值：训练集y
        self.y_train = self.h5rw.y_train
        
        # 赋值：测试集x
        self.x_test = self.h5rw.x_test
        
        # 赋值：测试集y
        self.y_test = self.h5rw.y_test

        # 赋值：x列表
        self.list_x = self.h5rw.list_x
        
        # 赋值：y列表
        self.list_y = self.h5rw.list_y

    def __init_dir(self):
        """ 初始化目录 """

        # 不含后缀的文件名
        self.filename = self.path_h5.stem

        # 模型存放目录
        self.dir_model = self.path_h5.parents[1] / 'model'

        # png存放目录
        self.dir_png = self.dir_model.parent / 'png'

        # 学习曲线png存放目录
        self.dir_png_lc = self.dir_png / 'lc'

        # 模型表现png存放目录
        self.dir_png_performance = self.dir_png / 'performance'

        # shap值相关png存放目录
        self.dir_png_shap = self.dir_png / 'shap'

        # 如果目录不存在，则创建目录
        for dir in [self.dir_model, self.dir_png, self.dir_png_lc, self.dir_png_performance, self.dir_png_shap]:
            if not dir.exists():
                dir.mkdir(parents=True)

        # 模型文件存放路径
        self.path_model = self.dir_model / f'{self.filename}_{suffix_kw}.skops'

    def fit(self):

        # print(f'██ Training... | Model: {suffix_kw} | N: {self.y_train.shape[0]}/{self.y_test.shape[0]} | {self.cv}-fold CV |')
        print(f'██ Training... | {self.filename} | {suffix_kw} | N: {self.y_train.shape[0]}/{self.y_test.shape[0]} | {self.cv}-fold CV | CPU: {self.cpu}')
   
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
            score = self.__tune_curve()

            # 交叉验证得分存入pd.DataFrame
            df_score = pd.DataFrame(data=score, index=todo_p)

            # 训练参数存入字典
            self.dict_params_all[self.current_p] = df_score

            # 最优参数存入字典
            if self.current_p in self.list_params_for_fitting:
                # 如果是n_estimators和max_depth，则拟合得到最优值
                result_p = self.__fitting_logistic()
                if result_p is not None:
                    self.dict_params_best[self.current_p] = self.__fitting_logistic()
                else:

                    # 从list中删除元素
                    self.list_params_for_fitting.remove(self.current_p)

                    # 非拟合数据存入字典
                    self.dict_params_best[self.current_p] = df_score['test_mean'].idxmax()
            else:
                self.dict_params_best[self.current_p] = df_score['test_mean'].idxmax()

        """ 生成最优模型，并预测训练集和测试集 """
        # 使用最优参数进行模型初始化
        self.model = self.__create_model(dict_param=self.dict_params_best)
        
        # 拟合
        self.model.fit(X=self.x_train, y=self.y_train)

        # 预测训练集和测试集
        self.__predict()

        """ 保存模型 """
        print('模型保存...', end=' ')

        sio.dump(obj=self.model, file=self.path_model, compression=zipfile.ZIP_LZMA, compresslevel=3)

        print('完成！')

        """ 保存训练超参数和模型表型 """
        print('学习曲线保存...', end=' ')

        self.h5rw.write_hyperparameters(model=self, group=suffix_kw)

        print('完成！')

        # 合并数组数据至pd.DataFrame、pd.Series
        self.__array2df()
        
        # 计算shap值
        self.__cal_shap()

        # 保存学习曲线图片
        self.plot_lc()

        # 保存模型表现图片
        self.plot_performance()

        # 保存shap值排序
        self.plot_shap_global()

    def __tune_curve(self):
        """ 获取学习曲线 """

        # 创建模型
        model = self.__create_model(dict_param=self.dict_params_j)

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

    def __create_model(self, dict_param: dict):
        """ 根据参数创建模型 """

        model = DecisionTreeRegressor(
            criterion=dict_param['criterion'],
            splitter=dict_param['splitter'],
            max_depth= dict_param['max_depth'],
            min_samples_split= dict_param['min_samples_split'],
            min_samples_leaf= dict_param['min_samples_leaf'],
            max_features=dict_param['max_features'],
            max_leaf_nodes=dict_param['max_leaf_nodes'],
            random_state=42,
        )

        return model

    def __fitting_logistic(self):
        """ 逻辑回归拟合 """
        
        # 拟合
        params, df = utils.fitting_logistic(self.dict_params_all[self.current_p].iloc[:, [0, 2]])
        
        # # 取train_mean的最大值
        # y_max_score = df.max(axis=0).iloc[1] * self.percent_max

        # 判断series是否全为nan
        # print('df_test_mean:\n', df['test_mean'])
        # if df['test_mean'].isna().all():
        #     return None

        # # 获得对应的x值
        # x_max_score = utils.function_logistic_inverse(y_max_score, *params.iloc[:, 1].to_numpy())
        x_max_score = utils.get_best_x(df.loc[:, 'test_mean'], self.percent_max)

        # 判断x_max_score是否为nan
        if np.isnan(x_max_score):
            return None

        # 拟合曲线存入字典
        self.dict_fitting[self.current_p] = df
        
        # 返回最优x和y
        return math.ceil(x_max_score)

    def __predict(self):
        """ 预测训练集和测试集 """

        # 预测训练集
        dict_predict_train = utils.predict(model=self.model, x=self.x_train, y=self.y_train)

        # 训练集模型表现
        self.r2_train, self.rmse_train, self.y_predict_train, self.mae_train = map(dict_predict_train.get, ['r2', 'rmse', 'predict', 'mae'])

        # 预测测试集
        dict_predict_test = utils.predict(model=self.model, x=self.x_test, y=self.y_test)

        self.r2_test, self.rmse_test, self.y_predict_test, self.mae_test = map(dict_predict_test.get, ['r2', 'rmse', 'predict', 'mae'])

    def __array2df(self):
        """ 合并数据 """
        
        # 合并训练集
        arr2d_train = np.hstack((self.x_train, self.y_train.reshape(self.x_train.shape[0], -1)))

        # 转化为DataFrame
        self.df_train = pd.DataFrame(
            data=arr2d_train,
            index=self.h5rw.index_train,
            columns=np.hstack((self.list_x, self.list_y)),
            )

        # 合并测试集
        arr2d_test = np.hstack((self.x_test, self.y_test.reshape(self.x_test.shape[0], -1)))

        # 转化为DataFrame
        self.df_test = pd.DataFrame(
            data=arr2d_test,
            index=self.h5rw.index_test,
            columns=np.hstack((self.list_x, self.list_y)),
        )

        # 合并训练集和测试集
        self.df_raw = pd.concat([self.df_train, self.df_test], axis=0)

        # 转换预测值为pd.DataFrame
        self.df_predict_train = pd.DataFrame(
            data=np.array([self.y_train, self.y_predict_train]).T,
            columns=['obs', 'predict'],
            index=self.df_train.index,
            )
        
        self.df_predict_test = pd.DataFrame(
            data=np.array([self.y_test, self.y_predict_test]).T,
            columns=['obs', 'predict'],
            index=self.df_test.index,
            )
        
        # print('self.df_train:\n', self.df_train)
        # print('self.df_test:\n', self.df_test)
        # print('self.df_raw:\n', self.df_raw)
        # print('self.df_predict_train:\n', self.df_predict_train)
        # print('self.df_predict_test:\n', self.df_predict_test)

    def __cal_shap(self):
        """ 计算SHAP值
        
        2023-06-19 v1
        多进程
        """

        print('计算及保存SHAP值...')

        # 调用函数计算
        self.df_shap, self.float_shap_expected_value, self.series_global_shap = generate_shap_values(
            model=self.model, x_df=self.df_raw.loc[:, self.list_x],
            n_jobs=self.cpu, tree_model=True, regression_model=True, boosting_model=False,
            )

        # 添加索引、设置索引名
        self.df_shap.index = self.df_raw.index
        self.df_shap.index.name = 'datetime'

        # 全局SHAP值设置feature列为索引
        self.series_global_shap = self.series_global_shap.set_index('feature', inplace=False).loc[:, 'shap_value']

        # 保存SHAP值
        self.h5rw.write_shap(model=self, group=suffix_kw)

        print('完成！')

    def plot_lc(self, show=False):
        """ 绘制学习曲线
            path_png: 图片保存路径, 如果指定则保存图片
            show: 是否显示图片
        
        2024-08-13 v1
        """

        # 作图顺序
        list_order = list(self.dict_params_all.keys())

        nrows = math.floor(math.sqrt(len(list_order)))
        ncols = math.ceil(len(list_order) / nrows)
        # print(nrows, ncols)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), dpi=100, sharey=False)
        ax = ax.flatten()

        for i, k in enumerate(list_order):
            
            ax[i].plot(self.dict_params_all[k].index, self.dict_params_all[k]['train_mean'], color='blue', label='train')
            ax[i].plot(self.dict_params_all[k].index, self.dict_params_all[k]['test_mean'], color='orange', label='test')

            ax[i].fill_between(self.dict_params_all[k].index, self.dict_params_all[k]['train_mean'] - self.dict_params_all[k]['train_std'], self.dict_params_all[k]['train_mean'] + self.dict_params_all[k]['train_std'], alpha=0.2)
            ax[i].fill_between(self.dict_params_all[k].index, self.dict_params_all[k]['test_mean'] - self.dict_params_all[k]['test_std'], self.dict_params_all[k]['test_mean'] + self.dict_params_all[k]['test_std'], alpha=0.2)
            ax[i].set_title(k)

            if k in self.list_params_for_fitting:
                ax[i].plot(self.dict_fitting[k].index, self.dict_fitting[k]['test_mean'], lw=1, label='fitting', color='red')
                ax[i].scatter(self.dict_params_best[k], self.dict_fitting[k].loc[self.dict_params_best[k], 'test_mean'], marker='o', s=100, label='best_params', color='red', zorder=10)
            else:
                ax[i].scatter(self.dict_params_best[k], self.dict_params_all[k].loc[self.dict_params_best[k], 'test_mean'], marker='o', s=100, label='best_params', color='red', zorder=10)

            ax[i].legend()
    
        plt.tight_layout()

        # 保存路径
        path_png = os.path.join(self.dir_png_lc, f'{self.filename}_{suffix_kw}.png')

        # 保存图片
        plt.savefig(path_png, dpi=300)
        
        # 显示图片
        if show:
            plt.show()
        else:
            plt.close()

    def plot_performance(self, show=False):
        """ 模型应用于测试集和验证集的表现 """
            
        # 画布设置
        fig, ax = plt.subplot_mosaic(
            mosaic=[
                    ['a', 'a', 'b'],
                    ['c', 'c', 'd'],
            ],
            # layout='constrained',
            # layout='tight',
            height_ratios=[1, 1],
            width_ratios=[4, 4, 4],
            figsize=(14, 8),
            # top=0.9,
        )

        # 训练数据及预测数据时间序列
        ax['a'].plot(self.df_predict_train.index, self.df_predict_train['obs'], color='grey', label='Observation', lw=0.5)
        ax['a'].plot(self.df_predict_train.index, self.df_predict_train['predict'], color='black', label='Prediction', lw=0.5)

        # 测试数据及预测数据时间序列
        ax['c'].plot(self.df_predict_test.index, self.df_predict_test['obs'], color='grey', label='Observation', lw=0.5)
        ax['c'].plot(self.df_predict_test.index, self.df_predict_test['predict'], color='black', label='Prediction', lw=0.5)

        # 标明train/test
        ax['a'].text(x=0.02, y=0.95, s='Training', color='black', ha='left', va='top', transform=ax['a'].transAxes, fontsize=20)
        ax['c'].text(x=0.02, y=0.95, s='Test', color='black', ha='left', va='top', transform=ax['c'].transAxes, fontsize=20)

        # xlabel、ylabel
        ax['a'].set_ylabel(self.list_y[0])
        ax['c'].set_ylabel(self.list_y[0])

        # 图例
        ax['a'].legend(loc='upper right', frameon=False, ncol=2)
        ax['c'].legend(loc='upper right', frameon=False, ncol=2)

        # 散点图和直方图
        plot.performance_scatter(
            data_={'r2': self.r2_train, 'rmse': self.rmse_train, 'df': self.df_predict_train, 'mae': self.mae_train},
            annotation_='Training',
            ax=ax['b'],
        )
        
        plot.performance_scatter(
            data_={'r2': self.r2_test, 'rmse': self.rmse_test, 'df': self.df_predict_test, 'mae': self.mae_test},
            annotation_='Test',
            ax=ax['d'],
        )

        plt.tight_layout()

        # 保存路径
        path_png = os.path.join(self.dir_png_performance, f'{self.filename}_{suffix_kw}.png')

        # 保存图片
        plt.savefig(path_png, dpi=300)
        
        # 显示图片
        if show:
            plt.show()
        else:
            plt.close()

    def plot_shap_global(self, show=False):
        """ global shapley value作图

        2023-06-25 v1
        """

        # 画布设置
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 10))

        # 准备数据
        series_name = self.series_global_shap.copy()

        # 排序
        # pd.Series().sort_values()
        series_name.sort_values(ascending=True, inplace=True)

        # 计算百分比
        series_name_percent = series_name / series_name.sum() * 100

        # 作图
        axn = ax.barh(y=series_name.index, width=series_name.values, color='#f42756', height=0.75)

        # xlabel
        ax.set_xlabel(r'$\overline{\mathrm{|shap|}}$')

        # 子图标题
        ax.set_title(suffix_kw)

        # 设置label
        labels = ['%.1f%%' % v for v in series_name_percent]
        # labels = ['%.2f/%.1f%%' % (series_name.iloc[i], series_name_percent.iloc[i]) for i in range(series_name.shape[0])]
        # labels = [f'{}/{series_name_percent.iloc[i]}:.1%' for i in range(series_name.shape[0])]
        ax.bar_label(container=axn, labels=labels, fmt='%.1f', label_type='edge')

        # xlim、ylim
        ax.set_xlim((0, ax.get_xlim()[1] * 1.15))
        ax.set_ylim((-0.75, series_name.shape[0] - 0.25))

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)

        # 保存路径
        path_png = os.path.join(self.dir_png_shap, f'{self.filename}_shap_global_{suffix_kw}.png')

        # 保存图片
        plt.savefig(path_png, dpi=300)
        
        # 显示图片
        if show:
            plt.show()
        else:
            plt.close()


class DT(ShapBasedExplainer):
    """ 决策树回归模型, 继承自ShapBasedExplainer, 用于替代DecisionTree类 """

    def __init__(self, path_h5: Path, cv: int = 5, cpu: int = 1):
        """ 初始化 """

        super().__init__(path_h5, cv, cpu)

        # 模型简称
        self.abbrname = 'dt'
        
        # 需要拟合的参数
        self.list_params_for_fitting = []

        # 对需要拟合的参数，寻找最优值时，取最大值的百分数
        self.percent_max = 0.98

        # 学习曲线初始参数
        self.dict_params_init = {
            'criterion': ['absolute_error', 'friedman_mse', 'poisson', 'squared_error'],
            'splitter': ['best', 'random'],
            'max_depth': [1, 2, 3, 4, 5, 7, 9, 12, 15, 20],
            'min_samples_split': [2, 5, 10, 20, 50],
            'min_samples_leaf': [1, 2, 5, 10, 20],
            'max_features': [0.2, 0.4, 0.6, 0.8, 1.0],
            'max_leaf_nodes': [1, 5, 10, 20],
        }

        # 初始参数-通常可以得到较的模型表现
        self.dict_params_overfitting = {
            'criterion': 'squared_error',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'max_leaf_nodes': None,
        }

        # 当前待优化参数的取值范围
        # self.todo_p = None

    def __create_model(self, dict_param: dict):
        """ 根据参数创建模型 """

        model = DecisionTreeRegressor(
            criterion=dict_param['criterion'],
            splitter=dict_param['splitter'],
            max_depth= dict_param['max_depth'],
            min_samples_split= dict_param['min_samples_split'],
            min_samples_leaf= dict_param['min_samples_leaf'],
            max_features=dict_param['max_features'],
            max_leaf_nodes=dict_param['max_leaf_nodes'],
            random_state=42,
        )

        return model

    def __tune_curve(self):
        """ 获取学习曲线 """

        # 创建模型
        model = self.__create_model(dict_param=self.dict_params_j)

        return utils.train_batch(
            model=model,
            X=self.x_train,
            y=self.y_train,
            cv=self.cv,
            cpu=self.cpu,
            param_name=self.current_p,
            param_range=self.todo_p,
        )

    def __fitting_logistic(self):
        """ 逻辑回归拟合 """
        
        # 拟合
        params, df = utils.fitting_logistic(self.dict_params_all[self.current_p].iloc[:, [0, 2]])
        
        # # 取train_mean的最大值
        # y_max_score = df.max(axis=0).iloc[1] * self.percent_max

        # 判断series是否全为nan
        # print('df_test_mean:\n', df['test_mean'])
        # if df['test_mean'].isna().all():
        #     return None

        # # 获得对应的x值
        # x_max_score = utils.function_logistic_inverse(y_max_score, *params.iloc[:, 1].to_numpy())
        x_max_score = utils.get_best_x(df.loc[:, 'test_mean'], self.percent_max)

        # 判断x_max_score是否为nan
        if np.isnan(x_max_score):
            return None

        # 拟合曲线存入字典
        self.dict_fitting[self.current_p] = df
        
        # 返回最优x和y
        return math.ceil(x_max_score)

    def plot_lc(self, show=False):
        """ 绘制学习曲线
            path_png: 图片保存路径, 如果指定则保存图片
            show: 是否显示图片
        
        2024-08-13 v1
        """

        # 作图顺序
        list_order = list(self.dict_params_all.keys())

        nrows = math.floor(math.sqrt(len(list_order)))
        ncols = math.ceil(len(list_order) / nrows)
        # print(nrows, ncols)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), dpi=100, sharey=False)
        ax = ax.flatten()

        for i, k in enumerate(list_order):
            
            ax[i].plot(self.dict_params_all[k].index, self.dict_params_all[k]['train_mean'], color='blue', label='train')
            ax[i].plot(self.dict_params_all[k].index, self.dict_params_all[k]['test_mean'], color='orange', label='test')

            ax[i].fill_between(self.dict_params_all[k].index, self.dict_params_all[k]['train_mean'] - self.dict_params_all[k]['train_std'], self.dict_params_all[k]['train_mean'] + self.dict_params_all[k]['train_std'], alpha=0.2)
            ax[i].fill_between(self.dict_params_all[k].index, self.dict_params_all[k]['test_mean'] - self.dict_params_all[k]['test_std'], self.dict_params_all[k]['test_mean'] + self.dict_params_all[k]['test_std'], alpha=0.2)
            ax[i].set_title(k)

            if k in self.list_params_for_fitting:
                ax[i].plot(self.dict_fitting[k].index, self.dict_fitting[k]['test_mean'], lw=1, label='fitting', color='red')
                ax[i].scatter(self.dict_params_best[k], self.dict_fitting[k].loc[self.dict_params_best[k], 'test_mean'], marker='o', s=100, label='best_params', color='red', zorder=10)
            else:
                ax[i].scatter(self.dict_params_best[k], self.dict_params_all[k].loc[self.dict_params_best[k], 'test_mean'], marker='o', s=100, label='best_params', color='red', zorder=10)

            ax[i].legend()
    
        plt.tight_layout()

        # 保存路径
        path_png = os.path.join(self.dir_png_lc, f'{self.filename}_{suffix_kw}.png')

        # 保存图片
        plt.savefig(path_png, dpi=300)
        
        # 显示图片
        if show:
            plt.show()
        else:
            plt.close()

    def fit(self):

        # print(f'██ Training... | Model: {suffix_kw} | N: {self.y_train.shape[0]}/{self.y_test.shape[0]} | {self.cv}-fold CV |')
        print(f'██ Training... | {self.filename} | {suffix_kw} | N: {self.y_train.shape[0]}/{self.y_test.shape[0]} | {self.cv}-fold CV | CPU: {self.cpu}')
   
        """ 依次对各个参数进行优化 """
        list_p = list(self.dict_params_init.keys())

        for i, self.current_p in enumerate(list_p):

            # 参数
            self.todo_p = self.dict_params_init[self.current_p]
            if not isinstance(self.todo_p, (list, np.ndarray)):
                # 参数储存至字典
                self.dict_params_best[self.current_p] = self.todo_p
                continue
                
            """ 学习曲线/交叉验证 """
            # 准备用于创建模型的参数
            self.dict_params_j = self.dict_params_best.copy()
            self.dict_params_j.update({self.current_p: self.todo_p[0]})
            self.dict_params_j.update({k: self.dict_params_overfitting[k] for k in list_p if k not in self.dict_params_j.keys()})

            # 准备用于print的参数
            self.dict_params_j2 = self.dict_params_j.copy()
            self.dict_params_j2[self.current_p] = self.todo_p

            print(f'调参({i+1}/{len(list_p)}): {self.current_p} | {self.dict_params_j2}')
            
            # 学习曲线
            score = self.__tune_curve()

            # 交叉验证得分存入pd.DataFrame
            df_score = pd.DataFrame(data=score, index=self.todo_p)

            # 训练参数存入字典
            self.dict_params_all[self.current_p] = df_score

            # 最优参数存入字典
            if self.current_p in self.list_params_for_fitting:
                # 如果是n_estimators和max_depth，则拟合得到最优值
                result_p = self.__fitting_logistic()
                if result_p is not None:
                    self.dict_params_best[self.current_p] = self.__fitting_logistic()
                else:

                    # 从list中删除元素
                    self.list_params_for_fitting.remove(self.current_p)

                    # 非拟合数据存入字典
                    self.dict_params_best[self.current_p] = df_score['test_mean'].idxmax()
            else:
                self.dict_params_best[self.current_p] = df_score['test_mean'].idxmax()

        """ 生成最优模型，并预测训练集和测试集 """
        # 使用最优参数进行模型初始化
        self.model = self.__create_model(dict_param=self.dict_params_best)
        
        # 拟合
        self.model.fit(X=self.x_train, y=self.y_train)

        # 预测训练集和测试集
        self._predict()

        """ 保存模型 """
        print('模型保存...', end=' ')

        sio.dump(obj=self.model, file=self.path_model, compression=zipfile.ZIP_LZMA, compresslevel=3)

        print('完成！')

        """ 保存训练超参数和模型表型 """
        print('学习曲线保存...', end=' ')

        self.h5rw.write_hyperparameters(model=self, group=suffix_kw)

        print('完成！')

        # 保存学习曲线图片
        self.plot_lc()

        # 保存模型表现图片
        self.plot_performance()


if __name__ == '__main__':
    pass
