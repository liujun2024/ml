""" 常用的一些函数 """
from __future__ import annotations
# import sys
import math
import numpy as np
import pandas as pd
from sklearn import metrics
# from scipy import stats
from scipy.optimize import curve_fit
from tkinter import Tk
from tkinter.filedialog import askdirectory
from pathlib import Path
from dask.distributed import Client


def askdir(initialdir='C:\\'):
    """ 获取用户选择的文件夹路径 """

    # 创建一个Tkinter窗口
    root = Tk()

    # 隐藏窗口
    root.withdraw()

    # 设置根窗口保持在最上层
    root.attributes('-topmost', True)

    # 打开文件夹选择对话框
    dir_project_ = askdirectory(initialdir=initialdir, title='请选择数据集*.h5文件所在目录...', parent=root)
    if not dir_project_:
        raise ValueError('未知路径！')
    
    return Path(dir_project_)


def split_data(data: pd.DataFrame, test_size: float, random_state: int, q: int, y: str=''):
    """ 划分数据集，简单随机或者分层抽样

    Parameters
    ----------
    data : pd.DataFrame，待划分的数据集
    test_size : 测试集占比
    random_state : 随机种子
    q : 分层抽样时数据集划分的层数，q=1时为不分层抽样

    Returns
    -------
    train_data : pd.DataFrame，训练集
    test_data : pd.DataFrame，测试集

    2025-11-17  v1  Create by LiuJun
    """

    from sklearn.model_selection import train_test_split

    if q == 1:
        # 不分层抽样
        
        return train_test_split(data, test_size=test_size, shuffle=True, random_state=random_state)

    elif q > 1:
        # 分层抽样
        y_bins = pd.qcut(data[y], q=q, labels=False)

        return train_test_split(data, test_size=test_size, shuffle=True, random_state=random_state, stratify=y_bins)

    else:
        raise ValueError('q must be a positive integer')


def predict(model, x: np.ndarray, y: np.ndarray):
    """ 使用随机森林模型进行预测

        path_skops: os.PathLike，模型文件的具体路径
        data: pd.DataFrame，待预测的数据，含有索引，最后一列为因变量，其它列为自变量

        return: {
            'r2': r2,
            'rmse': rmse,
            'predict': np.1darray,
        }

    2023-06-19 v1
    2024-07-19 v2
    单进程单线程
    """

    # 预测
    y_predict = model.predict(X=x)

    # 计算root mean squared error, RMSE 均方根误差
    rmse = metrics.root_mean_squared_error(y_true=y, y_pred=y_predict)

    # 计算mean absolute error, MAE平均绝对误差
    mae = metrics.mean_absolute_error(y_true=y, y_pred=y_predict)

    # computes the coefficient of determination, usually denoted as R2
    r2 = metrics.r2_score(y_true=y, y_pred=y_predict)

    dict_result = {
        'rmse': rmse,
        'r2': r2,
        'predict': y_predict,
        'mae': mae,
    }

    # 返回数据
    return dict_result


def fitting_logistic(data: pd.DataFrame | pd.Series):
    """ Logistic函数拟合, 参见Origin Basic Functions 
    
        data: 索引值作为x, 其它列的值分别作为y

    2023-01-08 v1
    单进程
    """
    
    # 类型识别, 如果是pd.Series, 先转化为pd.DataFrame
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, pd.Series):
        data = data.to_frame()
    else:
        raise TypeError("data must be pd.DataFrame or pd.Series")
    
    # 提取x值
    x_ = data.index.to_numpy()
    
    # 生成拟合后的曲线: x
    # x_fit = np.linspace(min(x_), max(x_), 100)
    x_fit = np.arange(min(x_), max(x_) + 1, 1)

    # 使用字典存储拟合参数和拟合曲线
    dict_params = dict()
    dict_curve = dict()

    # 依次拟合各列数据
    for c in data.columns:

        try:
            # 使用curve_fit进行拟合
            params_c, covariance_c = curve_fit(f=function_logistic, 
                                            xdata=x_, 
                                            ydata=data.loc[:, c].to_numpy(), 
                                            p0=[0, 1, 5, 3],
                                            bounds=([-1, 0, 0.1, 0], [0, 1, 1000, 10]),
                                            )
            # print('params_c:', params_c)
            # 生成拟合后的曲线: y
            y_fit = function_logistic(x_fit, *params_c)
        
        except RuntimeError:
            
            # 拟合无法收敛时，使用全nan填充
            params_c = np.full(4, np.nan)
            y_fit = np.full(x_fit.shape[0], np.nan)

        # 拟合参数存入字典
        dict_params[c] = params_c

        # 拟合曲线存入字典
        dict_curve[c] = y_fit

    # 合并拟合参数
    df_params = pd.DataFrame(dict_params, index=['A1', 'A2', 'x0', 'p'])

    # 合并拟合曲线
    df_curve = pd.DataFrame(dict_curve, index=x_fit)

    # 返回数据
    return df_params, df_curve


def function_logistic(x, A1, A2, x0, p):
    """ Logistic函数, 参见Origin Basic Functions """

    return A2 + (A1 - A2) / (1 + (x / x0) ** p)


def function_logistic_inverse(y, A1, A2, x0, p):
    """ 前面Logistic函数的反函数 """

    return x0 * math.exp(math.log((A1 - A2) / (y - A2) - 1) / p)


def get_best_x(series: pd.Series, percent_max: float):
    """ 获取最优参数 """

    # 归一化数据
    normalized_data = (series - series.min()) / (series.max() - series.min())
    # print('normalized_data:\n', normalized_data)

    # 计算与 self.percent_max 的差值
    diff = np.abs(normalized_data - percent_max)
    # print('diff:\n', diff)

    # 找到最小差值对应的索引
    index_closest = diff.idxmin()
    # print('index_closest:', index_closest)

    # 返回最优参数
    return index_closest


def train_batch(model, X, y, param_name, param_range, dask_client: Client | None = None, cpu=1, cv=10):
    """ 改变模型的某个参数，进行批量训练，返回学习曲线数据
    
    Parameters
    ----------
    model : 模型
    X : 自变量
    y : 因变量
    param_name : 待调参数名称
    param_range : 待调参数范围
    cv : 交叉验证次数
    dask_client : Dask 客户端，如果为 None，则使用本地训练

    Returns
    -------
    学习曲线数据
    {
        'train_mean': np.ndarray,
        'train_std': np.ndarray,
        'test_mean': np.ndarray,
        'test_std': np.ndarray,
    }

    2025-11-17  v1  Create by LiuJun
    """

    if dask_client is None:
        # 本地训练
        return train_batch_local(
            model=model, 
            X=X, 
            y=y, 
            param_name=param_name, 
            param_range=param_range, 
            cpu=cpu, 
            cv=cv,
        )

    else:
        # 分布式训练
        return train_batch_dask(
            model=model,
            X=X, 
            y=y, 
            param_name=param_name, 
            param_range=param_range, 
            dask_client=dask_client, 
            cpu=cpu, 
            cv=cv,
        )
    

def train_batch_local(model, X, y, param_name, param_range, cpu=1, cv=10):
    """ 改变模型的某个参数，进行批量训练，返回学习曲线数据
    
    Parameters
    ----------
    model : 模型
    X : 自变量
    y : 因变量
    param_name : 待调参数名称
    param_range : 待调参数范围
    cv : 交叉验证次数

    Returns
    -------
    学习曲线数据
    {
        'train_mean': np.ndarray,
        'train_std': np.ndarray,
        'test_mean': np.ndarray,
        'test_std': np.ndarray,
    }

    2025-08-13
    """

    from sklearn.model_selection import validation_curve

    # 使用validation_curve获取学习曲线数据
    score_train, score_test = validation_curve(
        estimator=model,
        X=X, 
        y=y,
        cv=cv,
        n_jobs=cpu,
        param_name=param_name, 
        param_range=param_range,
    )

    # 计算平均值和标准差
    dict_score = {
        'train_mean': np.mean(score_train, axis=1),
        'train_std': np.std(score_train, axis=1),
        'test_mean': np.mean(score_test, axis=1),
        'test_std': np.std(score_test, axis=1),
    }

    return dict_score


def train_batch_dask(model, X, y, param_name, param_range, dask_client: Client, cpu=1, cv=10):
    
    # 分布式训练
    print(f"🚀 在 Dask Worker 上执行...")
    

    def worker_task(X, y):
        """在 Dask Worker 上执行的函数"""

        from sklearn.model_selection import validation_curve
        
        score_train, score_test = validation_curve(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            n_jobs=-1,  # Worker 使用所有本地核心
            param_name=param_name,
            param_range=param_range,
        )
        
        return score_train, score_test


    # 预加载数据到 Worker（如果还没加载过）
    if not hasattr(dask_client, '_train_data_cached'):
        X_future = dask_client.scatter(X, broadcast=True)
        y_future = dask_client.scatter(y, broadcast=True)
        dask_client._train_data_cached = (X_future, y_future)
    else:
        X_future, y_future = dask_client._train_data_cached

    # 提交任务到 Worker
    future = dask_client.submit(worker_task, X_future, y=y_future)

    # 获取结果
    score_train, score_test = future.result()

    # 计算平均值和标准差
    dict_score = {
        'train_mean': np.mean(score_train, axis=1),
        'train_std': np.std(score_train, axis=1),
        'test_mean': np.mean(score_test, axis=1),
        'test_std': np.std(score_test, axis=1),
    }

    return dict_score 


if __name__ == '__main__':

    pass

    # 测试文件夹选择函数
    # 文件路径选择对话框
    # a = askdir()
    # print(a)

    # top = Tk()
    # top.withdraw()
    # print('ddd')
    # dir_project_ = askdirectory(initialdir='G:\\MLToolkit1.0_Project\\', title='请选择项目路径...')
    # if not dir_project_:
    #     raise ValueError('未知的项目路径！')
