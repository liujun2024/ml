

from pathlib import Path
import numpy as np
import regressor


if __name__ == '__main__':

    # h5文件路径
    path_h5 = Path(r'h5\DS1.h5')

    # cv\cpu
    cv = 5
    cpu = 8

    # 模型初始化
    model_rfr = regressor.RandomForest(path_h5=path_h5, cv=cv, cpu=cpu)

    # 参数初始化
    model_rfr.dict_params_init = {
                'n_estimators': [1, 2, 5, 10, 20, 50, 100, 200, 500],
                'max_depth': [1, 2, 5, 10, 20, 50, 100],
                'min_samples_split': np.arange(start=2, stop=21, step=2),
                'min_samples_leaf': np.arange(start=1, stop=20, step=4),
                'max_features': np.linspace(0.1, 1, 5),
                'max_samples': np.linspace(0.1, 1, 5),
        }

    # 训练
    model_rfr.fit()
