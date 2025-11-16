
import shap
import numpy as np
import pandas as pd
import skops.io as sio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def cal_shap_interaction_values(explainer: shap.TreeExplainer, data: pd.DataFrame):
    """ 计算SHAP interaction values，多进程计算的基函数

        explainer: shap.TreeExplainer
        data: pd.DataFrame, 仅包含x的数据

    2024-09-19 v1
    """

    return explainer.shap_interaction_values(data)


def cal_shap_interaction_values_mp(model:Path | shap.TreeExplainer, data: pd.DataFrame, n_jobs: int):
    """ 计算SHAP interaction values，多进程 

        model: Path or shap.TreeExplainer
        data: pd.DataFrame, 仅包含x的数据
        n_jobs: int, 进程数

    2024-09-19 v1
    多进程
    """

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
    pass
