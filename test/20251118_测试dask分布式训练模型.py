import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from auto_shap import generate_shap_values
from ml import _utils as utils
from dask.distributed import Client
from ml.regressor import RF
from ml import _hdf5 as hdf5
from ml import _utils as utils
from pathlib import Path


# connect the dask client
dask_client_address = 'tcp://172.16.156.6:8786'

if __name__ == '__main__':

    # h5文件保存路径
    dir_h5 = Path(__file__).parent / 'h5'
    # if not dir_h5.exists():
    #     dir_h5.mkdir(parents=True)

    # # 生成测试数据
    # X, y = make_regression(n_samples=10000, n_features=30, random_state=42)

    # # 存入dataframe
    # df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])], index=range(X.shape[0]))
    # df['y'] = y

    # # 切分数据集
    # df_train_bins, df_test_bins = utils.split_data(
    #     data=df, test_size=0.3, random_state=42, q=10, y='y'
    # )

    # # 存入h5文件
    # hdf5.raw2h5(
    #     df_train=df_train_bins,
    #     df_test=df_test_bins,
    #     labels=['y'],
    #     path_h5=dir_h5 / 'test01.h5',
    # )

    # 模型初始化
    model_rfr = RF(
        path_h5=dir_h5 / 'test01.h5',
        cv=5,
    )
    
    # 参数初始化
    model_rfr.dict_params_init = {
        'n_estimators': [1, 2, 5, 10, 20, 50, 100, 200, 500],
        'max_depth': [1, 2, 5, 10, 20, 50, 100],
        'min_samples_split': np.arange(start=2, stop=21, step=2),
        'min_samples_leaf': np.arange(start=1, stop=20, step=4),
        'max_features': np.linspace(0.1, 1, 5),
        'max_samples': np.linspace(0.1, 1, 5),
    }

    # 训练模型
    model_rfr.fit(cpu=8, dask_client_address=dask_client_address)

    # 计算shap值
    model_rfr.calculate_shap(cpu=8, dask_client_address=dask_client_address)
