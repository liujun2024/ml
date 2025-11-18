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


# connect the dask client
client = Client('tcp://172.16.156.6:8786')

if __name__ == '__main__':

    # 生成测试数据
    X, y = make_regression(n_samples=10000, n_features=30, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    test_data = X[:2000]  # 测试200个样本

    # numpy -> pandas
    test_data_pd = pd.DataFrame(test_data, columns=[f"f{i}" for i in range(test_data.shape[1])])

    print('test_data:', test_data_pd)

    # 方法1：本地计算
    start1 = time.time()
    # df_shap, shap_expected_values, shap_global_abs_avg = utils.calculate_shap_local(model=model, X=test_data_pd, cpu=4)
    start2 = time.time()

    # 方法2：基于代码2
    df_shap2, shap_expected_values2, shap_global_abs_avg2 = utils.calculate_shap_dask(model=model, X=test_data_pd, dask_client=client, cpu=4)
    start3 = time.time()

    print('1 elapsed:', start2 - start1)
    print('2 elapsed:', start3 - start2)

    # 结果对比
    # print('df_shap:', df_shap)
    print('df_shap2:', df_shap2)
    # print('shap_expected_values:', shap_expected_values)
    print('shap_expected_values2:', shap_expected_values2)
    # print('shap_global_abs_avg:', shap_global_abs_avg)
    print('shap_global_abs_avg2:', shap_global_abs_avg2)

