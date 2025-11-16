from allin1.ml import regressor, hdf5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# h5文件路径
path_h5 = Path(r'G:\_ING\_P05\ML\ML08\h5\ML08.h5')

# 读取h5文件
# h5 = hdf5.HDF5RW(path_h5)

# 载入模型
rf = regressor.RandomForest(
    path_h5=path_h5,
)

# 读取shap值
rf.h5rw.read_shap(group='rf')

# 赋值
rf.df_shap = rf.h5rw.df_shap
rf.series_global_shap = rf.h5rw.global_shap

print(rf.df_raw)
print(rf.h5rw.df_shap)
# print(rf.h5rw.global_shap)

# 绘图
rf.plot_shap_global1(show=True)


