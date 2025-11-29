""" 线性回归(Linear Regression)相关的类和函数 """

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import zipfile

import skops.io as sio
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import validation_curve

# from .. import _hdf5 as h5
from ml import hdf5, utils, plot
# from .. import _utils as utils
# from .. import _plot as plot

# import _hdf5 as h5
# import _utils as utils
# import _plot as plot

# 模型关键字缩写
suffix_kw = 'linear'


class Linear:

    def __init__(self, path_h5: os.PathLike, cv=5, cpu=4):

        # hdf5文件路径
        self.path_h5 = path_h5

        # 初始化目录
        self.__init_dir()

        # 交叉验证cv-fold
        self.cv = cv

        # 并行处理的cpu核心数
        self.cpu = cpu

        # 学习曲线初始参数
        self.dict_params_init = {'fit_intercept': [True, False]}

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

        # 读取hdf5文件训练集和测试集数据
        self.h5rw = hdf5.HDF5RW(path_h5=self.path_h5)

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


        """ 训练 """
        # 学习曲线
        score = self.__tune_curve()

        # 交叉验证得分存入pd.DataFrame
        df_score = pd.DataFrame(data=score, index=self.dict_params_init['fit_intercept'])

        # 训练参数存入字典
        self.dict_params_all['fit_intercept'] = df_score

        # 最优参数存入字典
        self.dict_params_best['fit_intercept'] = df_score['test_mean'].idxmax()

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
        
        # 保存学习曲线图片
        self.plot_lc()

        # 保存模型表现图片
        self.plot_performance()

    def __tune_curve(self):
        """ 获取学习曲线 """

        # 创建模型
        model = self.__create_model(dict_param=self.dict_params_init)

        # 使用validation_curve获取学习曲线数据
        score_train, score_test = validation_curve(
            estimator=model,
            X=self.x_train, 
            y=self.y_train,
            cv=self.cv,
            n_jobs=self.cpu,
            param_name='fit_intercept', 
            param_range=self.dict_params_init['fit_intercept'],
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
        """ 根据参数创建模型

            fit_intercept: bool, False时拟合过原点

            return: model

        2024-08-14 v1
        """

        # 建立模型
        model = LinearRegression(
            fit_intercept=dict_param['fit_intercept'],
            n_jobs=self.cpu,
            copy_X=True,
            positive=False,
            )

        # 返回数据
        return model

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

    def plot_lc(self, show=False):
        """ 绘制学习曲线
            path_png: 图片保存路径, 如果指定则保存图片
            show: 是否显示图片
        
        2024-08-14 v1
        """

        # 画布设置
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=100, sharey=False)

        ax.plot(self.dict_params_all['fit_intercept'].index, self.dict_params_all['fit_intercept']['train_mean'], color='blue', label='train')
        ax.plot(self.dict_params_all['fit_intercept'].index, self.dict_params_all['fit_intercept']['test_mean'], color='orange', label='test')

        ax.fill_between(self.dict_params_all['fit_intercept'].index, self.dict_params_all['fit_intercept']['train_mean'] - self.dict_params_all['fit_intercept']['train_std'], self.dict_params_all['fit_intercept']['train_mean'] + self.dict_params_all['fit_intercept']['train_std'], alpha=0.2)
        ax.fill_between(self.dict_params_all['fit_intercept'].index, self.dict_params_all['fit_intercept']['test_mean'] - self.dict_params_all['fit_intercept']['test_std'], self.dict_params_all['fit_intercept']['test_mean'] + self.dict_params_all['fit_intercept']['test_std'], alpha=0.2)
        ax.set_title('fit_intercept')

        ax.scatter(self.dict_params_best['fit_intercept'], self.dict_params_all['fit_intercept'].loc[self.dict_params_best['fit_intercept'], 'test_mean'], marker='o', s=100, label='best_params', color='red', zorder=10)

        ax.legend()
            
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

if __name__ == '__main__':
    pass
