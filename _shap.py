
import shap
import numpy as np
import pandas as pd
import skops.io as sio
from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor


# def cal_shap_interaction_values(explainer: shap.TreeExplainer, data: pd.DataFrame):
#     """ 计算SHAP interaction values，多进程计算的基函数

#         explainer: shap.TreeExplainer
#         data: pd.DataFrame, 仅包含x的数据

#     2024-09-19 v1
#     """

#     return explainer.shap_interaction_values(data)


# def cal_shap_interaction_values_mp(model: Path | shap.TreeExplainer, data: pd.DataFrame, n_jobs: int):
#     """ 计算SHAP interaction values，多进程 

#         model: Path or shap.TreeExplainer
#         data: pd.DataFrame, 仅包含x的数据
#         n_jobs: int, 进程数

#     2024-09-19 v1
#     多进程
#     """

#     # 模型初始化
#     if isinstance(model, Path):
#         model = sio.load(model)
#         explainer = shap.TreeExplainer(model=model)
#     elif isinstance(model, shap.TreeExplainer):
#         explainer = model
#     else:
#         raise ValueError('model must be Path or shap.TreeExplainer')

#     # 数据切分
#     list_data = np.array_split(data, n_jobs)

#     # 进程池初始化
#     pool = ProcessPoolExecutor(max_workers=n_jobs)

#     # 多进程计算
#     list_shap_interaction = [i for i in pool.map(cal_shap_interaction_values, [explainer]*n_jobs, list_data)]

#     # 合并结果
#     arr3d_result = np.vstack(list_shap_interaction)

#     return arr3d_result


def cal_shap_interactions(model: Path | shap.TreeExplainer, data: pd.DataFrame, cpu: int) -> dict:
    """ 计算SHAP interaction values
    
    Parameters
    ----------
    model : Path or shap.TreeExplainer
    data : pd.DataFrame, 仅包含x的数据
    cpu : int, 进程数

    2025-11-18  v1  Created by LiuJun，衍生自cal_shap_interaction_values和cal_shap_interaction_values_mp
    """

    from ml._utils import split_df

    def ca_joblib(explainer, data):
        return explainer.shap_interaction_values(data)

    # 特征列表
    list_x = data.columns.tolist()

    # 模型初始化
    if isinstance(model, Path):

        # load the model
        model = sio.load(
            file=model,
            trusted=[
                'xgboost.core.Booster', 
                'xgboost.sklearn.XGBRegressor',
                'collections.defaultdict', 
                'lightgbm.basic.Booster', 
                'lightgbm.sklearn.LGBMRegressor',
            ],  
        )

        # initialize the TreeExplainer
        explainer = shap.TreeExplainer(model=model)

    elif isinstance(model, shap.TreeExplainer):
        explainer = model

    else:
        raise ValueError('model must be Path or shap.TreeExplainer')

    if cpu == 1:
        # 单进程计算
        arr3d_result = explainer.shap_interaction_values(data)    # type: ignore
        # return explainer.shap_interaction_values(data)    # type: ignore
    else:

        # 开启joblib多进程计算
        from joblib import Parallel, delayed

        list_joblib = Parallel(n_jobs=cpu)(
            delayed(ca_joblib)(explainer, data_chunk) for data_chunk in split_df(data, cpu)
        )

        # 合并结果
        arr3d_result = np.concatenate(list_joblib, axis=0)      # type: ignore

        # 返回结果
        # return arr3d_result

    """ 计算平均绝对交互效应（MAIE）、交互作用占比（ITR） """
    # 结果存入字典
    # dict_main_effect = {}
    dict_maie = {}
    # dict_itr = {}

    # 遍历特征
    for x in list_x:

        # x在arr3d_result中的索引位置
        idx_x = list_x.index(x)

        # x与其它特征的交互强度存入字典
        dict_maie_x = {}
        # dict_itr_x = {}

        # 提取主效应并存入字典
        # dict_main_effect[x] = arr3d_result[:, idx_x, idx_x]

        # 计算x的主效应绝对值
        # arr1d_main_x = np.abs(arr3d_result[:, idx_x, idx_x])

        # 每个样本的总效应
        # arr1d_total_x = arr1d_main_x + np.sum(np.abs(arr3d_result[:, idx_x, :]), axis=1)

        # 分别计算其它特征i与x的平均绝对交互效应
        for i in range(arr3d_result.shape[1]):

            # 计算每个样本中x与特征i的交互效应
            arr1d_interaction_x_i = np.abs(arr3d_result[:, idx_x, i])

            # 计算绝对值平均
            maie = np.mean(arr1d_interaction_x_i)

            # 计算交互效应占比
            # itr = np.mean(arr1d_interaction_x_i / (arr1d_total_x + 1e-10))

            # 存入字典
            dict_maie_x[list_x[i]] = maie
            # dict_itr_x[list_x[i]] = itr
            
        # print(x, dict_itr_x)

        # 存入字典
        dict_maie[x] = dict_maie_x
        # dict_itr[x] = dict_itr_x
    
    # 字典转DataFrame
    df_maie = pd.DataFrame(dict_maie).T
    # df_itr = pd.DataFrame(dict_itr).T

    # columne顺序调整
    df_maie = df_maie[list_x]
    # df_itr = df_itr[list_x]

    # 为df_itr添加索引名，索引列为x，每列的值为其它特征与x交互作用在x total effect中的占比
    # df_itr.index.name = 'main'

    # df_itr.to_csv('itr.csv')
    # df_maie.to_csv('maie.csv')

    # print(df_maie)
    # print(df_itr)

    # 返回结果
    return {
        'arr3d_shap_interaction': arr3d_result,
        'df_maie': df_maie,
        # 'df_itr': df_itr,
    }


def cal_maie():
    """ 计算平均绝对交互效应（MAIE） """
    
    pass

    
if __name__ == '__main__':
    pass
