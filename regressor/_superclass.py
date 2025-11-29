""" 设置超类，用于继承 """

from __future__ import annotations
from typing import Literal
from abc import ABC, abstractmethod
import os
import math
import numpy as np
import pandas as pd
# import zipfile
from pathlib import Path
import skops.io as sio
import matplotlib.pyplot as plt
# from sklearn.model_selection import validation_curve
from auto_shap.auto_shap import generate_shap_values
import shap

from ml import hdf5, plot, utils

# from .. import _hdf5 as h5
# from .. import _plot as plot
# from .. import _utils as utils


class ShapBasedExplainer:
    """ 基于Shap值的解释器超类 """

    def __init__(self, path_h5: Path, cv: int = 5, cpu: int = 1):
        """ 初始化超类 """

        # hdf5文件路径
        self.path_h5 = path_h5

        # 模型简称：'rf', 'gbdt', 'xgboost', 'lgbm'
        self.abbrname : str = ''

        # 初始化目录
        self.__init_dir()

        # 交叉验证cv-fold
        self.cv = cv

        # 并行处理的cpu核心数
        self.cpu = cpu

        # 对需要拟合的参数，寻找最优值时，取最大值的百分数
        self.percent_max : float = 1

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
        self.dict_params_j = dict()

        # 当前正在训练的超参数
        self.current_p = None

        # 当前待优化参数的取值范围
        self.todo_p = None

        # 读取hdf5文件训练集和测试集数据
        self.h5rw = hdf5.HDF5RW(path_h5=self.path_h5)

        # 赋值：训练集x
        self.x_train : np.ndarray = self.h5rw.x_train
        
        # 赋值：训练集y
        self.y_train : np.ndarray = self.h5rw.y_train
        
        # 赋值：测试集x
        self.x_test : np.ndarray = self.h5rw.x_test
        
        # 赋值：测试集y
        self.y_test : np.ndarray = self.h5rw.y_test

        # 赋值：x列表
        self.list_x = self.h5rw.list_x
        
        # 赋值：y列表
        self.list_y = self.h5rw.list_y

        # 合并训练集
        arr2d_train : np.ndarray = np.hstack((self.x_train, self.y_train.reshape(self.x_train.shape[0], -1)))

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

        # shap值
        self.df_shap : pd.DataFrame

        # 载入模型
        self.__load_model()
    
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
        self.path_model = self.dir_model / f'{self.filename}_{self.abbrname}.skops'

    def __load_model(self):
        """ 载入模型 """

        # 模型不存在则直接返回
        if not self.path_model.exists():
            return
        
        # 载入模型
        self.model = sio.load(
            file=self.path_model,
            trusted=[
                'xgboost.core.Booster', 
                'xgboost.sklearn.XGBRegressor',
                'collections.defaultdict', 
                'lightgbm.basic.Booster', 
                'lightgbm.sklearn.LGBMRegressor',
            ]    
        )
    
        # 预测训练集和测试集
        self._predict()

    def _fitting_logistic(self):
        """ 逻辑回归拟合 """
        
        # 拟合
        params, df = utils.fitting_logistic(self.dict_params_all[self.current_p].iloc[:, [0, 2]])
        
        # # 取train_mean的最大值
        # y_max_score = df.max(axis=0).iloc[1] * self.percent_max

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

    def _predict(self):
        """ 预测训练集和测试集 """

        # 预测训练集
        dict_predict_train = utils.predict(model=self.model, x=self.x_train, y=self.y_train)

        # 训练集模型表现
        self.r2_train, self.rmse_train, self.y_predict_train, self.mae_train = map(dict_predict_train.get, ['r2', 'rmse', 'predict','mae'])

        # 预测测试集
        dict_predict_test = utils.predict(model=self.model, x=self.x_test, y=self.y_test)

        self.r2_test, self.rmse_test, self.y_predict_test, self.mae_test = map(dict_predict_test.get, ['r2', 'rmse', 'predict', 'mae'])
        
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
    
    def calculate_shap(self, cpu: int | None = None, overwrite=False, dask_client_address: str | None = None):
        """ 计算shap值, 用于替换auto_shap库 
        
        2025-11-18  v1  Created by LiuJun
        """

        print('计算SHAP值、保存、作图...', end=' ', flush=True)

        # 判断shap值是否已经存在
        if self.check_shap() and not overwrite:
            print('已存在，跳过！')
            return

        # 并行处理的cpu核心数
        if cpu is not None:
            self.cpu_shap = cpu
        
        # 判断使用使用dask分布式计算
        if dask_client_address is not None:
            from dask.distributed import Client
            client = Client(address=dask_client_address)

            self.df_shap, self.float_shap_expected_value, self.series_global_shap = utils.calculate_shap_dask(
                model=self.model, X=self.df_raw.loc[:, self.list_x],     # type: ignore
                dask_client=client, cpu=self.cpu_shap,
            )
        
        else:
            self.df_shap, self.float_shap_expected_value, self.series_global_shap = utils.calculate_shap_local(
                model=self.model, X=self.df_raw.loc[:, self.list_x],     # type: ignore
                cpu=self.cpu_shap,
            )

        # 保存SHAP值
        self.h5rw.write_shap(model=self, group=self.abbrname)

        # 保存shap值排序图
        self.plot_shap_global()

        # shap dependence图
        self.plot_shap_dependence()

        print('完成！')

    def calculate_shap_interaction(self, cpu: int=1, overwrite=False, dask_client_address: str | None = None):
        """ 计算SHAP interaction values
        
        2025-11-18  v1  Created by LiuJun 
        """

        from ml._shap import cal_shap_interactions

        print('计算SHAP interaction值、保存、作图...', end=' ', flush=True)

        # 开始计算
        # arr3d_shap_interaction = cal_shap_interactions(
        dict_shap_interaction = cal_shap_interactions(
            model=self.path_model,
            data=self.df_raw.loc[:, self.list_x],
            cpu=cpu,
        )

        # 保存
        self.h5rw.write_shap_interaction(data=dict_shap_interaction, group=self.abbrname)

        # 保存shap交互图
        self.plot_shap_interaction()

        print('完成！')

    def cal_shap(self, cpu: int | None = None, overwrite=False):
        """ 计算SHAP值 """

        print('计算SHAP值、保存、作图...', end=' ')

        # 判断shap值是否已经存在
        if self.check_shap() and not overwrite:
            print('已存在，跳过！')
            return

        # 并行处理的cpu核心数
        if cpu is not None:
            self.cpu_shap = cpu
        
        # 调用函数计算
        self.df_shap, self.float_shap_expected_value, self.series_global_shap = generate_shap_values(
            model=self.model, x_df=self.df_raw.loc[:, self.list_x],     # type: ignore
            n_jobs=self.cpu, tree_model=True, regression_model=True, boosting_model=True,
        )

        # 添加索引、设置索引名
        self.df_shap.index = self.df_raw.index
        self.df_shap.index.name = 'datetime'

        # 全局SHAP值设置feature列为索引
        self.series_global_shap = self.series_global_shap.set_index('feature', inplace=False).loc[:, 'shap_value']

        # 保存SHAP值
        self.h5rw.write_shap(model=self, group=self.abbrname)

        # 保存shap值排序图
        self.plot_shap_global()

        # shap dependence图
        self.plot_shap_dependence()

        print('完成！')

    def check_shap(self):
        """ 检查shap值是否存在 """
        
        # return self.h5rw.read_shap(group=self.abbrname)
        if self.h5rw.read_shap(group=self.abbrname) is not False:
            return True

    def read_shap(self):
        """ 读取shap值 """

        self.df_shap, self.float_shap_expected_value, self.series_global_shap = self.h5rw.read_shap(group=self.abbrname)

    def read_shap_interaction(self):
        """ 读取shap_interactions值 """

        self.arr3d_shap_interaction = self.h5rw.read_shap_interaction(group=self.abbrname)

    def plot_performance(self, show=False):
        """ 模型应用于测试集和验证集的表现 """

        # 画布设置
        _, ax = plt.subplot_mosaic(
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

        # 保存路径
        path_png = os.path.join(self.dir_png_performance, f'{self.filename}_{self.abbrname}.png')

        # 保存图片
        plt.savefig(path_png, dpi=100)
        
        # 显示图片
        if show:
            plt.show()
        else:
            plt.close()

    def plot_shap_global(self, show=False, dpi : int = 100):
        """ global shapley value作图

        2025.08.14 集成到ShapBasedExplainer类中
        """

        # 如果self.df_shap不存在则读取
        if 'df_shap' not in dir(self):
            self.read_shap()

        # 调用函数绘图
        plot.plotShapRanking(
            raw=self.df_raw,
            shap=self.df_shap,
            list_x=self.list_x,     # type: ignore
            path_png=self.dir_png_shap / f'{self.filename}_shap_global_{self.abbrname}.png',
            dpi=dpi,
            show=show,
        )

    def plot_shap_dependence(self, show=False, dpi : int = 100):
        """ shap dependence作图 """

        # 如果self.df_shap不存在则读取
        if 'df_shap' not in dir(self):
            self.read_shap()

        plot.shap_dependence(
            data_shap=self.df_shap,
            data_raw=self.df_raw,
            path_png=self.dir_png_shap / f'{self.filename}_shap_dependence_{self.abbrname}.png' if not show else None,
            dpi=dpi,
        )

    def plot_shap_interaction(self, show=False, dpi : int = 100):
        """ shap interaction作图 """

        # from matplotlib.axes import Axes

        # 如果self.arr3d_shap_interactions不存在则读取
        if 'arr3d_shap_interaction' not in dir(self):
            self.read_shap_interaction()

        # 如果self.df_shap不存在则读取
        if 'df_shap' not in dir(self):
            self.read_shap()

        # 提取根据重要性排序的特征列表
        # print(self.series_global_shap)
        list_x_sorted = self.series_global_shap.index.tolist()

        # 作图
        plot.shap_interaction_summary(
            arr3d_shap_interaction=self.arr3d_shap_interaction,     # type: ignore
            df_raw=self.df_raw,
            list_x=list_x_sorted,
            path_png=self.dir_png_shap / f'{self.filename}_shap_interaction_{self.abbrname}.png',
            dpi=dpi,
            show=show,
        )

        # """ 依次提取特征i与其它个特征的交互作用 """
        # dict_ij = {}
        # for i in range(len(list_x_sorted)):
        #     dict_i = {}
        #     for j in range(len(list_x_sorted)):
        #         dict_i[list_x_sorted[j]] = self.arr3d_shap_interaction[:, i, j]

        #     # 字典转DataFrame
        #     df_i = pd.DataFrame(dict_i)
        #     # df_i.columns = list_x_sorted
        #     df_i.index = self.df_shap.index
        #     print('df_i:\n', df_i)

        #     dict_ij[list_x_sorted[i]] = df_i

        # fig, axs = plt.subplots(figsize=(12, 8), dpi=dpi, nrows=1, ncols=len(list_x_sorted), layout='constrained')
        # axs : list[Axes] = axs.flatten()
        # print('axs:\n', axs)
    
        # for i, d in enumerate(list_x_sorted):

        #     if i == len(list_x_sorted) - 1:
        #         show_colorbar = True
        #     else:
        #         show_colorbar = False

        #     plot.beeswarm_base(
        #         df_raw=self.df_raw.loc[:, list_x_sorted],
        #         df_shap=dict_ij[d],
        #         ax=axs[i],
        #         # head=5,
        #         xlabel='',
        #         show_colorbar=show_colorbar,
        #         show_yticklabels=True if i == 0 else False,
        #         title=d,
        #     )

        #     # axs[i].set_title(d)

        # fig.supxlabel('SHAP interaction value')

        # plt.show()



if __name__ == '__main__':
    
    pass
