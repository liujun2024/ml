""" 随机森林相关的类和函数 

v0.1e: 将原有的使用logistic拟合更改为二次差值 updated 2024-10-12
"""

from __future__ import annotations
from typing import Literal
import warnings
import math
import numpy as np
import pandas as pd
import zipfile

from pathlib import Path
import skops.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import validation_curve
from auto_shap.auto_shap import generate_shap_values
from dask.distributed import Client


from .. import _hdf5 as h5
from .. import _utils as utils
from .. import _plot as plot
from .. import _shap as shap
from ._superclass import ShapBasedExplainer

# import _hdf5 as h5
# import _utils as utils
# import _plot as plot
# import _shap as shap


# 模型关键字缩写
suffix_kw = 'rf'


class RandomForest():
    """ 随机森林回归 """

    def __init__(self, path_h5: str | Path, cv=5, cpu=4, dask_client: Client | None = None):

        # 警告
        warnings.warn('此类将在后续版本中删除，请使用RF类代替！', DeprecationWarning)

        # hdf5文件路径
        self.path_h5 = Path(path_h5)

        # 初始化目录
        self.__init_dir()

        # 交叉验证cv-fold
        self.cv = cv

        # 并行处理的cpu核心数
        self.cpu = cpu

        # 计算shap值的cpu核心数
        self.cpu_shap = cpu

        # 需要拟合的参数
        self.list_params_for_fitting = ['n_estimators', 'max_depth']
        # self.list_params_for_fitting = ['n_estimators']

        # 对需要拟合的参数，寻找最优值时，取最大值的百分数
        self.percent_max = 0.995

        # 学习曲线初始参数
        self.dict_params_init = {
                    'n_estimators': [1, 2, 5, 10, 20, 50, 100, 200, 500],
                    'max_depth': [1, 2, 5, 10, 20, 50, 100],
                    'min_samples_split': np.arange(start=2, stop=21, step=2),
                    'min_samples_leaf': np.arange(start=1, stop=20, step=4),
                    'max_features': np.linspace(0.1, 1, 5),
                    'max_samples': np.linspace(0.1, 1, 5),
            }

        # 过拟合初始参数-通常可以得到很好的模型表现
        self.dict_params_overfitting = {
                'n_estimators': self.dict_params_init['n_estimators'][-1],
                'max_depth': self.dict_params_init['max_depth'][-1],
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'max_samples': 1.0,
        }

        # 当前待优化参数的取值范围
        self.todo_p = None

        # 用于储存最优参数，key为list_p中的元素，value为对应的最优值
        self.dict_params_best = dict()

        # 用于储存训练过程的所有结果，key为list_p中的元素，value为对应的pd.Series（index为参数值，data为对应的模型表现）
        self.dict_params_all = dict()
    
        # 用于储存交叉验证n_estimators和max_depth的拟合曲线
        self.dict_fitting = dict()

        # 阈值，默认为：0.005，用于判断n_estimators对应的模型表现衰减量，如果超过阈值，则需要继续添加n_estimators值进行迭代
        # self.threshold = 0.005

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

        # 训练集、测试集预测数据
        self.df_predict_train = None
        self.df_predict_test = None

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

        # 载入模型
        self.__load_model()

        # 使用dask分布式客户端
        self.dask_client = dask_client
        self.use_dask = dask_client is not None

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

    def __load_model(self):
        """ 如果模型存在，则先载入模型 """

        # 模型不存在则直接返回
        if not self.path_model.exists():
            return
        
        # 载入模型
        self.model = sio.load(self.path_model)
    
        # 预测训练集和测试集
        self.__predict()

    # def fit(self):
    #     """ 改造后的fit方法，支持分布式 """

    #     print(f'██ Training... | {self.filename} | {suffix_kw} | N: {self.y_train.shape[0]}/{self.y_test.shape[0]} | {self.cv}-fold CV | CPU: {self.cpu}')
        
    #     if self.use_dask:
    #         # 使用分布式训练
    #         self.__fit_distributed()
    #     else:
    #         # 使用本地训练
    #         self.__fit_local()

    def fit(self):
        
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
            
            # 学习曲线调参
            score = self.__tune_curve()

            # 交叉验证得分存入pd.DataFrame
            df_score = pd.DataFrame(data=score, index=self.todo_p)

            # 最优参数存入字典
            if self.current_p in self.list_params_for_fitting:

                # 寻找第1次学习曲线中的max_score，将其归一化
                df_score['test_mean_normalized'] = df_score['test_mean'] / df_score['test_mean'].max()
                # print('df_score:\n', df_score)

                # 判断阈值在哪两个score中间，寻找对应的位置索引
                # index_insert = df_score['test_mean_normalized'].searchsorted(self.percent_max)
                index_insert = find_first_index(df_score['test_mean_normalized'].tolist(), self.percent_max)
                # print('index_insert:', index_insert)
                # print('a:', a)

                # print('self.percent_max:', self.percent_max)

                # 阈值所在区间的score范围
                score_left = df_score.index[index_insert - 1]
                score_right = df_score.index[index_insert]

                # 在score_left和score_right之间，均匀插入5个整数
                self.todo_p = np.unique(np.linspace(score_left, score_right, 7, dtype=int, endpoint=True)[1:-1])

                self.dict_params_j2[self.current_p] = self.todo_p
                print(f'调参({i+1}/{len(list_p)}): {self.current_p} | {self.dict_params_j2}')
                
                # 第2次学习曲线调参
                score2 = self.__tune_curve()

                # 交叉验证得分存入pd.DataFrame
                df_score2 = pd.DataFrame(data=score2, index=self.todo_p)

                # 合并第1次和第2次学习曲线的结果
                df_score = pd.concat([df_score.iloc[:, :-1], df_score2], axis=0)

                # 去除重复行
                df_score = df_score[~df_score.index.duplicated(keep='first')]

                # 重新索引
                df_score.sort_index(inplace=True, ascending=True)

                # 归一化max_score
                df_score['test_mean_normalized'] = df_score['test_mean'] / df_score['test_mean'].max()

                # 寻找与self.percent_max最接近的score
                p_final = (df_score['test_mean_normalized'] - self.percent_max).abs().idxmin()

                # 结果存入字典
                self.dict_params_best[self.current_p] = p_final

            else:
                self.dict_params_best[self.current_p] = df_score['test_mean'].idxmax()
            
            # 训练参数存入字典
            # print('dfscore:\n', df_score)
            self.dict_params_all[self.current_p] = df_score

        """ 生成最优模型，并预测训练集和测试集 """
        # 使用最优参数进行模型初始化
        self.model = self.__create_model(dict_param=self.dict_params_best)
        
        # 拟合
        self.model.fit(X=self.x_train, y=self.y_train)

        """ 保存模型 """
        print('模型保存...', end=' ')

        sio.dump(obj=self.model, file=self.path_model, compression=zipfile.ZIP_LZMA, compresslevel=3)

        print('完成！')

        # 预测训练集和测试集
        self.__predict()

        """ 保存训练超参数和模型表型 """
        print('学习曲线数据保存...', end=' ')

        self.h5rw.write_hyperparameters(model=self, group=suffix_kw)

        print('完成！')

        # 保存学习曲线图片
        self.plot_lc()

        # 保存模型表现图片
        self.plot_performance()

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
            param_range=self.todo_p,    # # v0.1e中更改
            dask_client_address=self.dask_client, 
        )

    def __create_model(self, dict_param: dict):
        """ 根据参数创建模型 """

        # 建立随机森林回归模型
        model = RandomForestRegressor(
            n_estimators=dict_param['n_estimators'],  # 决策树数量
            criterion="squared_error",  # criterion：'squared_error' = mse, 'absolute_error' = mae, 'poisson'
            max_depth=dict_param['max_depth'],  # None：默认最大深度，最高复杂度
            min_samples_split=dict_param['min_samples_split'],  # 节点样本数量小于2时不再分支，最高复杂度
            min_samples_leaf=dict_param['min_samples_leaf'],  # 叶子节点所包含的最小样本数，最高复杂度
            min_weight_fraction_leaf=0.0,  # 0：不同样本间的权重一致
            max_features=dict_param['max_features'],  # max_features：'sqrt', 'log2', None}, int or float, default=1.0即max_features=n_features，最高复杂度
            max_leaf_nodes=None,  # 不限制叶子节点的个数
            min_impurity_decrease=0.0,  #
            bootstrap=True,  # https://stackoverflow.com/questions/40131893/random-forest-with-bootstrap-false-in-scikit-learn-python
            # If bootstrap is True, the number of samples to draw from X to train each base estimator， 默认最高复杂度
            oob_score=False,  #
            n_jobs=self.cpu,  #
            random_state=42,  #
            verbose=0,  # 控制台输出信息丰富程度
            warm_start=False,  #
            ccp_alpha=0.0,  #
            max_samples=dict_param['max_samples'],
        )

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

    def cal_shap(self, cpu: int | None = None):
        """ 计算SHAP值
        
        2023-06-19 v1
        多进程
        """

        print('计算SHAP值、保存、作图...')

        # 载入模型
        # model = sio.load(self.path_model)

        if cpu is not None:
            self.cpu_shap = cpu

        # 根据self.dask_client确定是否使用分布式计算
        if self.use_dask:

            # 预加载数据
            model_future = self.dask_client.scatter(self.model, broadcast=True)
            x_df_future = self.dask_client.scatter(self.df_raw.loc[:, self.list_x], broadcast=True)

            # 提交任务
            future = self.dask_client.submit(
                generate_shap_values,
                model=model_future, 
                x_df=x_df_future,      # type: ignore
                # n_jobs=self.cpu_shap, 
                n_jobs=1, 
                tree_model=True, 
                regression_model=True,
            )

            # 等待任务完成并获取结果
            self.df_shap, self.float_shap_expected_value, self.series_global_shap = future.result()

        else:

            # 调用函数计算
            self.df_shap, self.float_shap_expected_value, self.series_global_shap = generate_shap_values(
                model=self.model, x_df=self.df_raw.loc[:, self.list_x],      # type: ignore
                n_jobs=self.cpu_shap, tree_model=True, regression_model=True,
            )

        # 添加索引、设置索引名
        self.df_shap.index = self.df_raw.index
        self.df_shap.index.name = 'datetime'

        # 全局SHAP值设置feature列为索引
        self.series_global_shap = self.series_global_shap.set_index('feature', inplace=False).loc[:, 'shap_value']

        print(f'self.df_shap:\n{self.df_shap}')
        print(f'self.series_global_shap:\n{self.series_global_shap}')

        # 保存SHAP值
        self.h5rw.write_shap(model=self, group=suffix_kw)

        # 保存shap值排序图
        self.plot_shap_global()

        # shap dependence图
        self.plot_shap_dependence()

        print('完成！')

    def check_shap(self):
        """ 检查shap值是否存在 """
        
        return self.h5rw.read_shap(group='rf')

    def __cal_shap_interaction(self):
        """ 计算SHAP interaction values """
        
        print('计算SHAP interaction值、保存、作图...')

        # 调用函数
        arr3d_shap_interaction = shap.cal_shap_interaction_values_mp(
            model=self.path_h5,
            data=self.df_raw.loc[:, self.list_x],
            n_jobs=self.cpu,
        )
        
        # 保存
        self.h5rw.write_shap_interaction(data=arr3d_shap_interaction, group=suffix_kw)

        # 保存shap值排序图
        # self.plot_shap_global()

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
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), dpi=100, sharey=False, layout='constrained')
        ax = ax.flatten()

        for i, k in enumerate(list_order):
            
            # 子图标题
            ax[i].set_title(k)
            
            # 线图
            ax[i].plot(self.dict_params_all[k].index, self.dict_params_all[k]['train_mean'], color='blue', label='train')
            ax[i].plot(self.dict_params_all[k].index, self.dict_params_all[k]['test_mean'], color='orange', label='test')

            # 误差范围填充
            ax[i].fill_between(self.dict_params_all[k].index, self.dict_params_all[k]['train_mean'] - self.dict_params_all[k]['train_std'], self.dict_params_all[k]['train_mean'] + self.dict_params_all[k]['train_std'], alpha=0.2)
            ax[i].fill_between(self.dict_params_all[k].index, self.dict_params_all[k]['test_mean'] - self.dict_params_all[k]['test_std'], self.dict_params_all[k]['test_mean'] + self.dict_params_all[k]['test_std'], alpha=0.2)

            # 最优参数标记
            # print('dd', self.dict_params_best[k])
            # print('ee', self.dict_params_all[k].loc[self.dict_params_best[k], 'test_mean'])
            ax[i].scatter(self.dict_params_best[k], self.dict_params_all[k].loc[self.dict_params_best[k], 'test_mean'], marker='o', s=100, label='best', color='red', zorder=10)
            
            # x轴log
            if k in self.list_params_for_fitting:
                ax[i].set_xscale('log')

            # legend
            ax[i].legend(frameon=False)
    
        # plt.tight_layout()

        # 保存路径
        # path_png = os.path.join(self.Dir_png_lc, f'{self.filename}_{suffix_kw}.png')
        path_png = self.dir_png_lc / f'{self.filename}_{suffix_kw}.png'

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

        # plt.tight_layout()

        # 保存路径
        # path_png = os.path.join(self.Dir_png_performance, f'{self.filename}_{suffix_kw}.png')
        Path_png = self.dir_png_performance / f'{self.filename}_{suffix_kw}.png'

        # 保存图片
        plt.savefig(Path_png, dpi=100)
        
        # 显示图片
        if show:
            plt.show()
        else:
            plt.close()

    def plot_shap_global(self, show=False):
        """ global shapley value作图

        2023.06.25  Created by LiuJun
        2025.06.03  新增beeswarm图
        """

        # from allin1.ml._plot_beeswarm import summary_legacy
        # from allin1.ml._colors import red_blue
        from ml._plot_beeswarm import summary_legacy
        from ml._colors import red_blue

        # 复制可变对象，避免修改原始数据
        df_shap = self.df_shap.copy()

        # 计算全局shap值
        df_shap_global = df_shap.abs().mean(axis=0)

        # 升序排列
        df_shap_global.sort_values(ascending=True, inplace=True)

        # 画布设置
        _, (ax1, ax2) = plt.subplots(figsize=(8, 10), ncols=2, nrows=1, dpi=100, sharey=False, layout='constrained')

        # 特征重要性
        df_shap_global.plot.barh(width=0.7, color='#1e87e4', ax=ax1, zorder=10)

        # 网格线
        ax1.grid(visible=True, which='major', axis='y', color="#cccccc", lw=0.5, dashes=(1, 4), zorder=0)

        # 做beeswarm图，并获得Axes对象
        ax_ = summary_legacy(
            shap_values=df_shap.to_numpy(), 
            features=self.df_raw.loc[:, self.list_x].to_numpy(),      # type: ignore
            feature_names=df_shap.columns.tolist(),
            max_display=df_shap.shape[1],
        )

        # 将beeswarm图的PathCollection对象添加到子图ax2中
        ax2.add_collection(ax_.collections[0])

        # x轴标题设置
        ax1.set_xlabel('mean(|SHAP value|)', verticalalignment='center_baseline')
        ax2.set_xlabel('SHAP value', verticalalignment='center_baseline')

        # 删除ax2的y轴刻度标签
        ax2.set_yticklabels([])

        # 对齐ax1和ax2的y轴范围
        ax2.set_ylim(ax1.get_ylim())

        # ax2图的colorbar
        ax_cbar = ax2.inset_axes(bounds=(1.02, 0, 0.03, 1))

        # 使用预设的colormap
        m = cm.ScalarMappable(cmap=red_blue)

        # 设置刻度
        m.set_array([0, 1])

        # 绘制colorbar
        cb = plt.colorbar(m, cax=ax_cbar, extend='neither', ticks=[0.01, 0.99])

        # 刻度值
        cb.set_ticklabels(['low', 'high'])
        
        # 隐藏边框
        cb.outline.set_visible(False)   # type: ignore
        
        # 刻度值大小
        cb.ax.tick_params(labelsize=14, length=0)

        # colorbar标题
        # cb.set_label('Feature value', size=14, labelpad=-12, va='bottom')
        cb.set_label('Feature value', labelpad=-12, va='bottom')
        # cb.set_alpha(1)

        # 调整子图间距为0
        plt.tight_layout()
        plt.subplots_adjust(wspace=0)

        # 保存图片
        plt.savefig(self.dir_png_shap / f'{self.filename}_shap_global_{suffix_kw}.png', dpi=100)
        
        # 显示图片
        if show:
            plt.show()

    def plot_shap_dependence(self, show=False):
        """ shap dependence作图

        2025.06.05  Created by LiuJun
        """

        plot.shap_dependence(
            data_shap=self.df_shap,
            data_raw=self.df_raw,
            path_png=self.dir_png_shap / f'{self.filename}_shap_dependence_{suffix_kw}.png' if not show else None,
        )


class RF(ShapBasedExplainer):
    """ 随机森林回归模型, 继承自ShapBasedExplainer, 用于替代RandomForest类 """

    from dask.distributed import Client

    def __init__(self, path_h5: Path, cv: int = 5):
        """ 初始化 """

        super().__init__(path_h5, cv)

        # 模型简称
        self.abbrname = 'rf'
        
        # 需要拟合的参数
        self.list_params_for_fitting = ['n_estimators', 'max_depth']

        # 对需要拟合的参数，寻找最优值时，取最大值的百分数
        self.percent_max = 0.995

        # 学习曲线初始参数
        self.dict_params_init = {
                    'n_estimators': [1, 2, 5, 10, 20, 50, 100, 200, 500],
                    'max_depth': [1, 2, 5, 10, 20, 50, 100],
                    'min_samples_split': np.arange(start=2, stop=21, step=2),
                    'min_samples_leaf': np.arange(start=1, stop=20, step=4),
                    'max_features': np.linspace(0.1, 1, 5),
                    'max_samples': np.linspace(0.1, 1, 5),
            }

        # 过拟合初始参数-通常可以得到很好的模型表现
        self.dict_params_overfitting = {
                'n_estimators': self.dict_params_init['n_estimators'][-1],
                'max_depth': self.dict_params_init['max_depth'][-1],
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'max_samples': 1.0,
        }

        # 当前待优化参数的取值范围
        # self.todo_p = None

    def __create_model(self, dict_param: dict):
        """ 根据参数创建模型 """

        # 建立随机森林回归模型
        model = RandomForestRegressor(
            n_estimators=dict_param['n_estimators'],  # 决策树数量
            criterion="squared_error",  # criterion：'squared_error' = mse, 'absolute_error' = mae, 'poisson'
            max_depth=dict_param['max_depth'],  # None：默认最大深度，最高复杂度
            min_samples_split=dict_param['min_samples_split'],  # 节点样本数量小于2时不再分支，最高复杂度
            min_samples_leaf=dict_param['min_samples_leaf'],  # 叶子节点所包含的最小样本数，最高复杂度
            min_weight_fraction_leaf=0.0,  # 0：不同样本间的权重一致
            max_features=dict_param['max_features'],  # max_features：'sqrt', 'log2', None}, int or float, default=1.0即max_features=n_features，最高复杂度
            max_leaf_nodes=None,  # 不限制叶子节点的个数
            min_impurity_decrease=0.0,  #
            bootstrap=True,  # https://stackoverflow.com/questions/40131893/random-forest-with-bootstrap-false-in-scikit-learn-python
            # If bootstrap is True, the number of samples to draw from X to train each base estimator， 默认最高复杂度
            oob_score=False,  #
            # n_jobs=self.cpu,  #
            random_state=42,  #
            verbose=0,  # 控制台输出信息丰富程度
            warm_start=False,  #
            ccp_alpha=0.0,  #
            max_samples=dict_param['max_samples'],
        )

        return model

    def __tune_curve(self, cpu: int = 1, dask_client_address: str | None = None):
        """ 获取学习曲线 """

        # 创建模型
        model = self.__create_model(dict_param=self.dict_params_j)

        return utils.train_batch(
            model=model,
            X=self.x_train,
            y=self.y_train,
            cv=self.cv,
            cpu=cpu,
            param_name=self.current_p,
            param_range=self.todo_p,
            dask_client_address=dask_client_address,
        )

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
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), dpi=100, sharey=False, layout='constrained')
        ax = ax.flatten()

        for i, k in enumerate(list_order):
            
            # 子图标题
            ax[i].set_title(k)
            
            # 线图
            ax[i].plot(self.dict_params_all[k].index, self.dict_params_all[k]['train_mean'], color='blue', label='train')
            ax[i].plot(self.dict_params_all[k].index, self.dict_params_all[k]['test_mean'], color='orange', label='test')

            # 误差范围填充
            ax[i].fill_between(self.dict_params_all[k].index, self.dict_params_all[k]['train_mean'] - self.dict_params_all[k]['train_std'], self.dict_params_all[k]['train_mean'] + self.dict_params_all[k]['train_std'], alpha=0.2)
            ax[i].fill_between(self.dict_params_all[k].index, self.dict_params_all[k]['test_mean'] - self.dict_params_all[k]['test_std'], self.dict_params_all[k]['test_mean'] + self.dict_params_all[k]['test_std'], alpha=0.2)

            # 最优参数标记
            # print('dd', self.dict_params_best[k])
            # print('ee', self.dict_params_all[k].loc[self.dict_params_best[k], 'test_mean'])
            ax[i].scatter(self.dict_params_best[k], self.dict_params_all[k].loc[self.dict_params_best[k], 'test_mean'], marker='o', s=100, label='best', color='red', zorder=10)
            
            # x轴log
            if k in self.list_params_for_fitting:
                ax[i].set_xscale('log')

            # legend
            ax[i].legend(frameon=False)
    
        # plt.tight_layout()

        # 保存路径
        # path_png = os.path.join(self.Dir_png_lc, f'{self.filename}_{suffix_kw}.png')
        path_png = self.dir_png_lc / f'{self.filename}_{suffix_kw}.png'

        # 保存图片
        plt.savefig(path_png, dpi=300)
        
        # 显示图片
        if show:
            plt.show()
        else:
            plt.close()

    def fit(self, cpu: int = 1, dask_client_address: str | None = None):

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
            
            # 学习曲线调参
            score = self.__tune_curve(cpu=cpu, dask_client_address=dask_client_address)

            # 交叉验证得分存入pd.DataFrame
            df_score = pd.DataFrame(data=score, index=self.todo_p)

            # 最优参数存入字典
            if self.current_p in self.list_params_for_fitting:

                # 寻找第1次学习曲线中的max_score，将其归一化
                df_score['test_mean_normalized'] = df_score['test_mean'] / df_score['test_mean'].max()
                # print('df_score:\n', df_score)

                # 判断阈值在哪两个score中间，寻找对应的位置索引
                # index_insert = df_score['test_mean_normalized'].searchsorted(self.percent_max)
                index_insert = find_first_index(df_score['test_mean_normalized'].tolist(), self.percent_max)
                # print('index_insert:', index_insert)
                # print('a:', a)

                # print('self.percent_max:', self.percent_max)

                # 阈值所在区间的score范围
                score_left = df_score.index[index_insert - 1]
                score_right = df_score.index[index_insert]

                # 在score_left和score_right之间，均匀插入5个整数
                self.todo_p = np.unique(np.linspace(score_left, score_right, 7, dtype=int, endpoint=True)[1:-1])

                self.dict_params_j2[self.current_p] = self.todo_p
                print(f'调参({i+1}/{len(list_p)}): {self.current_p} | {self.dict_params_j2}')
                
                # 第2次学习曲线调参
                score2 = self.__tune_curve(cpu=cpu, dask_client_address=dask_client_address)

                # 交叉验证得分存入pd.DataFrame
                df_score2 = pd.DataFrame(data=score2, index=self.todo_p)

                # 合并第1次和第2次学习曲线的结果
                df_score = pd.concat([df_score.iloc[:, :-1], df_score2], axis=0)

                # 去除重复行
                df_score = df_score[~df_score.index.duplicated(keep='first')]

                # 重新索引
                df_score.sort_index(inplace=True, ascending=True)

                # 归一化max_score
                df_score['test_mean_normalized'] = df_score['test_mean'] / df_score['test_mean'].max()

                # 寻找与self.percent_max最接近的score
                p_final = (df_score['test_mean_normalized'] - self.percent_max).abs().idxmin()

                # 结果存入字典
                self.dict_params_best[self.current_p] = p_final

            else:
                self.dict_params_best[self.current_p] = df_score['test_mean'].idxmax()
            
            # 训练参数存入字典
            # print('dfscore:\n', df_score)
            self.dict_params_all[self.current_p] = df_score

        """ 生成最优模型，并预测训练集和测试集 """
        # 使用最优参数进行模型初始化
        self.model = self.__create_model(dict_param=self.dict_params_best)
        
        # 拟合
        self.model.fit(X=self.x_train, y=self.y_train)

        """ 保存模型 """
        print('模型保存...', end=' ')

        sio.dump(obj=self.model, file=self.path_model, compression=zipfile.ZIP_LZMA, compresslevel=3)

        print('完成！')

        # 预测训练集和测试集
        self._predict()

        """ 保存训练超参数和模型表型 """
        print('学习曲线数据保存...', end=' ')

        self.h5rw.write_hyperparameters(model=self, group=suffix_kw)

        print('完成！')

        # 保存学习曲线图片
        self.plot_lc()

        # 保存模型表现图片
        self.plot_performance()


def find_first_index(lst: list, b) -> int:
    """ 按照顺序查找第一个大于等于b的元素的索引 """

    return next((i for i, num in enumerate(lst) if b <= num), -1)


if __name__ == "__main__":
    pass