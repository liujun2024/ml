""" 算法包括: GBDT, XGBoost, LightGBM, RF, LR """


# import os
from pathlib import Path
import shap
import numpy as np
import regressor
# from PyQt6.QtWidgets import QFileDialog, QApplication
# import _hdf5 as h5
import _utils as utils


# app = QApplication([])

if __name__ == '__main__':

    # 

    # 选择待读取的*.txt文件所在目录
    # dir_h5 = QFileDialog.getExistingDirectory(None, "选择待读取的*.h5文件所在目录", "./")
    Dir_h5 = utils.askdir('./')
    # dir_h5 = r'G:\_ING\O3SOA\h5'

    # if not Dir_h5:
    #     exit(0)

    # Dir_h5 = Path(Dir_h5)

    # cv\cpu
    cv = 10
    cpu = 8

    # 逐一训练
    # for file in os.listdir(dir_h5):
    for path_h5 in Dir_h5.iterdir():

        print(path_h5.name)

        if path_h5.suffix != '.h5':
            continue

        # 模型文件保存路径
        # path_model = path_h5.parents[1] / 'model' / f'{path_h5.stem}_dt.skops'
        # if path_model.exists():
        #     continue

        # 模型初始化
        # model_gbdt = regressor.GradientBoostingDecisionTree(path_h5=path_h5, cv=cv, cpu=cpu)
        # model_xgb = regressor.ExtremeGradientBoosting(path_h5=path_h5, cv=cv, cpu=cpu)
        # model_lgb = regressor.LightGradientBoostingMachine(path_h5=path_h5, cv=cv, cpu=cpu)
        # model_rfr = regressor.RandomForest(path_h5=path_h5, cv=cv, cpu=cpu)
        model_dt = regressor.DecisionTree(path_h5=path_h5, cv=cv, cpu=cpu)
        # model_lr = regressor.Linear(path_h5=path_h5, cv=cv, cpu=cpu)

        # 初始参数
        # model_gbdt.dict_params_init = {
        #     'n_estimators': [2, 5, 10, 20, 50, 200],
        #     'max_depth': [1, 2, 3, 4, 5, 7, 9, 12, 15, 20],
        #     'min_samples_split': [2, 5, 10, 20, 50],
        #     'min_samples_leaf': [1, 2, 5, 10, 20],
        #     'max_features': [0.2, 0.4, 0.6, 0.8, 1.0],
        # }

        # model_xgb.dict_params_init = {
        #     'n_estimators': [1, 2, 5, 10, 20, 50],
        #     'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 20],
        #     'max_bin': [10, 20, 50, 100, 150, 256],
        # }

        # model_lgb.dict_params_init = {
        #     'num_leaves': [20, 30, 60, 100, 200],
        #     'n_estimators': [1, 2, 5, 10, 20, 50],
        #     'min_child_samples': [10, 20, 50, 100], 
        # }

        # model_rfr.dict_params_init = {
        #             'n_estimators': [1, 2, 5, 10, 20, 50, 200],
        #             'max_depth': [1, 2, 5, 10, 20, 50, 100],
        #             'min_samples_split': np.arange(start=2, stop=21, step=2),
        #             'min_samples_leaf': np.arange(start=1, stop=20, step=4),
        #             'max_features': np.linspace(0.1, 1, 5),
        #             'max_samples': np.linspace(0.1, 1, 5),
        #     }

        # 多模型依次训练
        # for model in [model_lgb, model_gbdt, model_rfr, model_xgb, model_lr]:
        # for model in [model_gbdt, model_rfr, model_xgb, model_lr]:
        for model in [model_dt]:

            # 训练
            model.fit()
            # model.__cal_shap()
