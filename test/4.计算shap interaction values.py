
import time
from pathlib import Path
import numpy as np
import pandas as pd
import regressor
import shap
import skops.io as sio
from auto_shap import generate_shap_values
import _hdf5 as hdf5





if __name__ == '__main__':

    # skops文件路径
    path_model = Path(r'model\DS1_rf.skops')

    # h5文件路径
    path_h5 = Path(r'h5\DS1.h5')

    # 读取x数据
    file = hdf5.HDF5RW(path_h5=path_h5)

    df_x = pd.DataFrame(data=file.x_train, columns=file.list_x, index=file.index_train).iloc[:100, :]
    print('x_train:\n', df_x)

    # cpu
    cpu = 4

    # 载入模型
    model = sio.load(path_model)

    print('model:', model)
    t0 = time.time()

    # 计算shap值
    explainer = shap.TreeExplainer(model=model) 
    df_shap = explainer.shap_values(df_x, approximate=False, check_additivity=False)
    df_shap = pd.DataFrame(data=df_shap, columns=df_x.columns, index=df_x.index)

    t1 = time.time()
    print('df_shap:\n', df_shap)

    # 计算interaction values
    df_shap_interaction = explainer.shap_interaction_values(df_x)
    
    t2 = time.time()
    print('df_shap_interaction:\n', df_shap_interaction)

    print('shap:', t1-t0)
    print('shap interaction:', t2-t1)
    