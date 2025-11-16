
import time
from pathlib import Path
import numpy as np
import pandas as pd
import regressor
import shap
import skops.io as sio
from auto_shap import generate_shap_values
import _hdf5 as hdf5
from concurrent.futures import ProcessPoolExecutor


def cal_shap_interaction_values(explainer: shap.TreeExplainer, data: pd.DataFrame):
    """ 计算SHAP interaction values，多进程计算的基函数

        explainer: shap.TreeExplainer
        data: pd.DataFrame, 仅包含x的数据

    2024-09-19 v1
    """

    return explainer.shap_interaction_values(data)


def cal_shap_interaction_values_mp(model:Path | shap.TreeExplainer, data: pd.DataFrame, n_jobs: int):
    """  """

    # 模型初始化
    if isinstance(model, Path):
        model = sio.load(model)
        explainer = shap.TreeExplainer(model=model)
    elif isinstance(model, shap.TreeExplainer):
        explainer = model
    else:
        raise ValueError('model must be Path or shap.TreeExplainer')

    # 数据切分
    list_data = np.array_split(data, n_jobs)

    # 进程池初始化
    pool = ProcessPoolExecutor(max_workers=n_jobs)

    # 多进程计算
    list_shap_interaction = [i for i in pool.map(cal_shap_interaction_values, [explainer]*n_jobs, list_data)]

    # 合并结果
    arr3d_result = np.vstack(list_shap_interaction)

    return arr3d_result


if __name__ == '__main__':

    # skops文件路径
    path_model = Path(r'model\DS1_rf.skops')

    # h5文件路径
    path_h5 = Path(r'h5\DS1.h5')

    # 读取x数据
    file = hdf5.HDF5RW(path_h5=path_h5)

    df_x = pd.DataFrame(data=file.x_train, columns=file.list_x, index=file.index_train).iloc[:20, :]
    print('x_train:\n', df_x)

    # cpu
    cpu = 4

    # 调取函数
    arr3d_result = cal_shap_interaction_values_mp(model=path_model, data=df_x, n_jobs=cpu)

    print('df_shap_interaction:\n', arr3d_result)

    # 保存至h5文件
    file.write_shap_interaction(data=arr3d_result, group='rf')

    print('done')
