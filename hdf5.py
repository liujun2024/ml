""" hdf5文件读取相关的类和函数 """

from __future__ import annotations
import os
# import sys
import h5py
from pathlib import Path
# import time
# import config as cfg
import numpy as np
import pandas as pd


class HDF5RW:

    """  
    path_h5: str, hdf5文件路径
    index_type: str, 如果hdf5文件中不包含索引类型时生效, 
                默认为'datetime', 可选: "sequence"
    
    """

    def __init__(self, path_h5: str | Path, index_type='datetime'):

        # hdf5文件路径
        self.path_h5 = path_h5
        print('self.path_h5:', self.path_h5)

        # 索引类型
        self.index_type = index_type

        # 声明各变量类型
        self.global_shap : pd.Series = None

        # 判断h5文件是否存在
        if os.path.exists(self.path_h5):

            # 读取原始数据
            self.read_raw()

            # 读取模型表现数据
            # self.read_performance()

    def read_raw(self):

        # 打开h5文件
        self.f = h5py.File(name=self.path_h5, mode='r')

        if '_raw' not in self.f.keys():
            self.f.close()
            raise ValueError(f'{self.path_h5}中不存在raw数据！')

        # 读取训练数据和独立验证数据
        self.x_train : np.ndarray = self.f['_raw/x_train'][()]
        self.y_train : np.ndarray = self.f['_raw/y_train'][()]
        self.x_test : np.ndarray = self.f['_raw/x_test'][()]
        self.y_test : np.ndarray = self.f['_raw/y_test'][()]

        # 读取索引
        self.index_train : np.ndarray = self.f['_raw/index_train'][()]
        self.index_test : np.ndarray = self.f['_raw/index_test'][()]
        self.index_all : np.ndarray = np.hstack((self.index_train, self.index_test))

        if self.index_type == 'datetime':
            self.index_train : pd.DatetimeIndex = pd.to_datetime(self.index_train, unit='s')
            self.index_test: pd.DatetimeIndex = pd.to_datetime(self.index_test, unit='s')
        
        # 特征x和标签y列表
        self.list_x : list[str] = self.f['_raw'].attrs['features']
        self.list_y : list[str] = self.f['_raw'].attrs['labels']
        # print('self.list_y: ', self.list_y)

        # 如果标签y的个数为1，则直接将y_train和y_test转为1维数组
        if len(self.list_y) == 1:
            self.y_train = self.y_train[:, 0]
            self.y_test = self.y_test[:, 0]

        # 判断是否存在索引类型关键字
        if 'index_type' in self.f['_raw'].attrs:
            self.index_type = self.f['_raw'].attrs['index_type']
        else:
            self.index_type = self.index_type

        # 合并训练集
        arr2d_train = np.hstack((self.x_train, self.y_train.reshape(self.x_train.shape[0], -1)))

        # 转化为DataFrame
        self.df_train = pd.DataFrame(
            data=arr2d_train,
            index=self.index_train,
            columns=np.hstack((self.list_x, self.list_y)),
            )

        # 合并测试集
        arr2d_test = np.hstack((self.x_test, self.y_test.reshape(self.x_test.shape[0], -1)))

        # 转化为DataFrame
        self.df_test = pd.DataFrame(
            data=arr2d_test,
            index=self.index_test,
            columns=np.hstack((self.list_x, self.list_y)),
        )

        # 索引名
        self.df_train.index.name = self.index_type
        self.df_test.index.name = self.index_type

        # 合并训练集和测试集
        self.df_raw = pd.concat([self.df_train, self.df_test], axis=0)

        # 关闭文件
        self.f.close()

    def read_shap(self, group: str):
        """ 读取shap值 """

        # shap数据位置
        loc_shap = f'{group}/importance/shap'

        # 打开h5文件
        f = h5py.File(name=self.path_h5, mode='r')

        if loc_shap not in f.keys():
            f.close()
            raise ValueError(f'{self.path_h5}中不存在{group}的shap数据！')
            # return False

        # 读取数据并存入DataFrame，这里的columns之前有误，2025-01-03更正
        self.df_shap = pd.DataFrame(
            data=f[loc_shap][()],
            index=pd.to_datetime(self.index_all, unit='s'),
            columns=self.list_x,
            # columns=f[loc_shap].attrs['features'], # 错误项，
        )

        # 设置索引名称
        self.df_shap.index.name = 'datetime'

        # 读取全局重要性
        self.global_shap = pd.Series(
            data=f[loc_shap].attrs['global_shap'],
            index=f[loc_shap].attrs['features'],
            name='shap',
        )

        # 设置索引名称
        self.global_shap.index.name = 'features'

        # 读取shap_expected_value
        self.shap_expected_value = f[loc_shap].attrs['expected_value']

        # 关闭文件
        f.close()

        return (self.df_shap, self.shap_expected_value, self.global_shap)


    def write_hyperparameters(self, model, group: str):

        """ 保存调参数据至hdf5文件 
            model: 模型实例
            group: str, hdf5文件group名称
        2024-08-13 v1
        """

        # print('学习曲线保存...', end=' ')

        # 保存训练参数
        self.f = h5py.File(self.path_h5, 'a')

        # 删除group
        if group in self.f.keys():
            del self.f[group]

        # 保存超参数调参结果
        for k, v in model.dict_params_all.items():
            self.f[f'{group}/hyperparameters/{k}'] = v.to_numpy()
            self.f[f'{group}/hyperparameters/{k}'].attrs['index'] = v.index.tolist()
            self.f[f'{group}/hyperparameters/{k}'].attrs['columns'] = v.columns.tolist()
        
        # 保存任务类型
        self.f[group].attrs['task_type'] = 'regression'

        # 保存超参数作图顺序
        self.f[f'{group}/hyperparameters'].attrs['sequence_for_plot'] = list(model.dict_params_all.keys())

        # 保存预测结果
        self.f[f'{group}/predict/train'] = model.y_predict_train
        self.f[f'{group}/predict/train'].attrs['r2'] = model.r2_train
        self.f[f'{group}/predict/train'].attrs['rmse'] = model.rmse_train
        self.f[f'{group}/predict/train'].attrs['mae'] = model.mae_train
        
        self.f[f'{group}/predict/test'] = model.y_predict_test
        self.f[f'{group}/predict/test'].attrs['r2'] = model.r2_test
        self.f[f'{group}/predict/test'].attrs['rmse'] = model.rmse_test
        self.f[f'{group}/predict/test'].attrs['mae'] = model.mae_test

        self.f.close()

        # print('完成！')

    def read_performance(self):
        """ 读取h5文件中模型表现数据 """
        
        # t0 = time.time()

        # 打开h5文件
        self.f = h5py.File(name=self.path_h5, mode='r')

        # 模型列表
        self.list_model = [i for i in self.f.keys() if i != '_raw']

        if len(self.list_model) == 0:
            self.f.close()
            # raise ValueError(f'{self.path_h5}中无模型训练结果！')

        # print('model:', self.list_model)

        # 为每个模型创建一个字典，训练结果存入其中
        self.dict_model = {i: {} for i in self.list_model}

        # 读取原始数据
        # self.read_raw()

        # 读取各模型训练结果数据
        for model in self.list_model:
            
            # 超参数列表
            list_hyperparameters = list(self.f[f'{model}/hyperparameters'].keys())

            # 遍历超参数
            for hp in list_hyperparameters:

                # 读取超参数，写入字典
                self.dict_model[model][hp] = pd.DataFrame(
                    data=self.f[f'{model}/hyperparameters/{hp}'][()], 
                    columns=self.f[f'{model}/hyperparameters/{hp}'].attrs['columns'],
                    index=self.f[f'{model}/hyperparameters/{hp}'].attrs['index'],
                    )   # type: ignore

            # 读取作图顺序
            self.dict_model[model]['sequence_for_plot'] = self.f[f'{model}/hyperparameters'].attrs['sequence_for_plot']

            # 读取预测结果
            self.dict_model[model]['predict_train'] = self.f[f'{model}/predict/train'][()]  # type: ignore
            self.dict_model[model]['predict_test'] = self.f[f'{model}/predict/test'][()]    # type: ignore

            # 读取预测性能: r2, rmse
            self.dict_model[model]['r2_train'] = self.f[f'{model}/predict/train'].attrs['r2']
            self.dict_model[model]['rmse_train'] = self.f[f'{model}/predict/train'].attrs['rmse']
            self.dict_model[model]['mae_train'] = self.f[f'{model}/predict/train'].attrs['mae']
            
            self.dict_model[model]['r2_test'] = self.f[f'{model}/predict/test'].attrs['r2']
            self.dict_model[model]['rmse_test'] = self.f[f'{model}/predict/test'].attrs['rmse']
            self.dict_model[model]['mae_test'] = self.f[f'{model}/predict/test'].attrs['mae']

            # # 读取预测性能：mae
            # if 'mae' in self.f[f'{model}/predict/train'].attrs:
            #     self.dict_model[model]['mae_train'] = self.f[f'{model}/predict/train'].attrs['mae']
            #     self.dict_model[model]['mae_test'] = self.f[f'{model}/predict/test'].attrs['mae']
            # else:


        # 关闭文件
        self.f.close()

        # t1 = time.time()
        # print('读取h5文件完成，耗时：', round(t1 - t0, 2), '秒')

    def write_shap(self, model, group: str):
        """ SHAP值写入hdf5文件 
        
        2024-08-15 v1
        """
        
        # 打开hdf5文件
        f = h5py.File(name=self.path_h5, mode='a')

        # 目标dataset名称
        ds_name = f'{group}/importance/shap'

        # 如果dataset已存在，则删除
        if ds_name in f:
            del f[ds_name]

        # 写入shap_values_df
        f.create_dataset(name=ds_name, data=model.df_shap.to_numpy(), shuffle='T', compression='gzip', compression_opts=5)

        # 写入属性：global_shap_df、shap_expected_value
        f[ds_name].attrs['features'] = model.series_global_shap.index.to_numpy()
        f[ds_name].attrs['global_shap'] = model.series_global_shap.to_numpy()
        f[ds_name].attrs['expected_value'] = model.float_shap_expected_value

        # 关闭文件
        f.close()

    def write_shap_interaction(self, data: dict, group: str):
        """ SHAP interaction values写入hdf5文件 
            data: 3维numpy数组，shape=(n_samples, n_features, n_features)
            group: h5文件中的组名

        2024-09-19 v1
        """

        # 打开hdf5文件
        f = h5py.File(name=self.path_h5, mode='a')

        # 目标dataset名称
        ds_name = f'{group}/importance/shap_interaction'
        ds_name_maie = f'{group}/importance/shap_maie'

        # 如果dataset已存在，则删除
        if ds_name in f:
            del f[ds_name]
        
        if ds_name_maie in f:
            del f[ds_name_maie]

        # 写入shap_interaction_values, 3维数组
        f.create_dataset(name=ds_name, data=data['arr3d_shap_interaction'], shuffle='T', compression='gzip', compression_opts=5)

        # 写入绝对平均交互值MAIE
        f.create_dataset(name=ds_name_maie, data=data['df_maie'].to_numpy(), shuffle='T', compression='gzip', compression_opts=5)

        # 写入属性：features
        f[ds_name].attrs['features'] = data['df_maie'].columns.to_numpy()
        f[ds_name_maie].attrs['features'] = data['df_maie'].columns.to_numpy()

        # 关闭文件
        f.close()

    def read_shap_interaction(self, group: str):
        """ 读取SHAP interaction values 
        
        2025-11-18  v1  Created by LiuJun
        """
        
        # shap数据位置
        loc_shap_interaction = f'{group}/importance/shap_interaction'

        # 打开h5文件
        f = h5py.File(name=self.path_h5, mode='r')

        if loc_shap_interaction not in f.keys():
            f.close()
            return False
            # raise ValueError(f'{self.path_h5}中不存在{group}的shap数据！')

        # 读取3维数组
        arr3d_shap_interaction = f[loc_shap_interaction][()]    # type: ignore

        # 关闭文件
        f.close()

        return arr3d_shap_interaction



def raw2h5(df_train: pd.DataFrame, df_test: pd.DataFrame, labels: list, path_h5: Path, dict_attrs={}):
    """ 将原始数据保存至h5文件
        df_train: 训练集数据，pd.DataFrame，表头需与df_test一致
        df_test: 测试集数据，pd.DataFrame
        labels: 标签列表，即因变量列表；如果只有一个因变量，仍需传入列表，如：['y']
        dict_attrs: 待写入的属性字典，如：{'description': 'time resolution: hourly;'}
        path_h5: h5文件路径
    
    2024-08-15 v1
    """

    # 判断路径上级目录是否存在，不存在则创建
    if not Path(path_h5).parent.exists():
        Path(path_h5).parent.mkdir(parents=True)

    # 判断df_train和df_test的表头是否一致
    if df_train.columns.tolist() != df_test.columns.tolist():
        raise ValueError('df_train和df_test的表头不一致！')
    
    # 判断df_train的表头中是否有重复项
    if df_train.shape[1] != len(set(df_train.columns.tolist())):
        raise ValueError('df_train的表头中有重复项！')

    # 特征列表，即自变量列表
    list_x = [i for i in df_train.columns.tolist() if i not in labels]

    # 提取训练集和测试集数据
    x_train = df_train.loc[:, list_x]
    y_train = df_train.loc[:, labels]
    x_test = df_test.loc[:, list_x]
    y_test = df_test.loc[:, labels]

    # x_train = df_train.iloc[:, :-1]
    # y_train = df_train.iloc[:, -1]
    # x_test = df_test.iloc[:, :-1]
    # y_test = df_test.iloc[:, -1]

    # 提取特征和标签
    # list_x = df_all.columns.tolist()[:-1]
    # list_y = df_all.columns.tolist()[-1]

    # 新建h5文件，存在则替换
    # f = h5py.File(name=path_hdf5, mode='w')
    f = h5py.File(name=path_h5, mode='a')

    # 写入训练集和测试集数据
    f.create_dataset(name='_raw/x_train', data=x_train.to_numpy(), shuffle='T', compression='gzip', compression_opts=5)
    f.create_dataset(name='_raw/y_train', data=y_train.to_numpy(), shuffle='T', compression='gzip', compression_opts=5)
    f.create_dataset(name='_raw/x_test', data=x_test.to_numpy(), shuffle='T', compression='gzip', compression_opts=5)
    f.create_dataset(name='_raw/y_test', data=y_test.to_numpy(), shuffle='T', compression='gzip', compression_opts=5)

    # 判断索引类型
    if isinstance(df_train.index, pd.DatetimeIndex):
        """ 时间索引 """

        # 写入索引类型
        f['_raw'].attrs['index_type'] = 'datetime'

        # 整理索引数据，转换为秒级时间戳
        index_train = df_train.index.to_numpy().astype(np.uint64) * 1E-9
        index_test = df_test.index.to_numpy().astype(np.uint64) * 1E-9

    else:
        """ 数值索引 """

        # 写入索引类型
        f['_raw'].attrs['index_type'] = 'integer'

        # 整理索引数据
        index_train = df_train.index.to_numpy()
        index_test = df_test.index.to_numpy()
    
    # 写入索引数据
    f.create_dataset(name='_raw/index_train', data=index_train, shuffle='T', compression='gzip', compression_opts=5)
    f.create_dataset(name='_raw/index_test', data=index_test, shuffle='T', compression='gzip', compression_opts=5)

    # 写入属性：特征列表和标签列表
    f['_raw'].attrs['features'] = list_x
    f['_raw'].attrs['labels'] = labels

    # 判断dict_attrs字典是否有数据
    if dict_attrs:

        # 写入属性：其它信息
        for k, v in dict_attrs.items():
            f['_raw'].attrs[k] = v

    # f['_raw'].attrs['description'] = f'time resolution: hourly; site:{sitecode};'

    # 关闭文件
    f.close()


if __name__ == '__main__':

    pass
