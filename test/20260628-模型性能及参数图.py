from collections.abc import Iterable
from pathlib import Path
from ml import hdf5
from pprint import pprint
from smogchamber import kit
import matplotlib.pyplot as plt
from typing import Literal
from matplotlib.axes import Axes
from plot import base
import matplotlib.ticker as mticker


def plot_summary(
        dir_project: Path,
        name_prefix: Iterable[str], 
        figsize: tuple = (16, 12),
        group: str = 'rf',
        ncols: Literal[1, 2, 4] = 2,
        markersize_mean: int = 5,
        markersize_median: int = 3,
        rotation_x: int = 0,
):
    """

    
    """

    base.set_global_rcparams()

    # h5文件所在路径
    dir_hdf5 = dir_project / 'h5'

    # 存放有数据的name
    valid_names = []

    # 数据存入列表
    list_r2 = []    # 存入r2
    list_rmse = []  # 存入rmse
    list_mae = []   # 存入mae
    list_cv_r2 = [] # 存入交叉验证R2数组，用于box图
    list_residual_err = []  # 存入残差数组，用于box图
    list_slope = [] # 存入验证集斜率，Predict vs. Obs.

    # 遍历模型名称
    for name in name_prefix:

        # 获取模型名称对应的h5文件
        match_name = [f for f in dir_hdf5.glob(f'{name}*.h5') if f.is_file()]
        if match_name:
            path_name = match_name[0]
        else:
            print(f'名称未匹配：{name}')
            continue
        
        # 数据读取实例化
        h5_name = hdf5.HDF5RW(path_h5=path_name)
        
        # 读取模型性能数据
        status = h5_name.read_performance()
        if not status:
            continue

        dict_model = h5_name.dict_model

        if group not in dict_model:
            print(f'未发现{group}模型数据')
        
        # 获得模型数据
        data_name = dict_model[group]
        
        # 存入R2、RMSE、MAE、cv-r2
        list_r2.append(data_name['r2_test'])
        list_rmse.append(data_name['rmse_test'])
        list_mae.append(data_name['mae_test'])
        list_cv_r2.append(data_name['cv-r2'])

        # 获得predict_test
        arr1d_pred_test = data_name['predict_test']

        # 获得obs_test
        arr1d_obs_test = h5_name.y_test

        # 计算残差
        arr1d_residual_err = arr1d_pred_test - arr1d_obs_test
        list_residual_err.append(arr1d_residual_err)

        # 计算斜率
        (slope, intercept), _ = kit.fitting_equation(x=arr1d_obs_test, y=arr1d_pred_test, fitting_order=1)
        list_slope.append(slope)
        
        # 名称存入列表
        valid_names.append(name)

    # 准备画布
    fig, axs = plt.subplots(ncols=ncols, nrows=4 // ncols, figsize=figsize, layout='constrained', sharex=True)
    axs: list[Axes] = axs.flatten()

    # 作图：R2、RMSE、MAE
    a = axs[0].plot(list_r2, marker='o', label='R$^2$', color='tab:blue')
    axs0_right = axs[0].twinx()
    b = axs0_right.plot(list_rmse, marker='s', label='RMSE', color='tab:orange')
    c = axs0_right.plot(list_mae, marker='^', label='MAE', color='tab:green')

    axs[0].set_ylabel('R$^2$')
    axs[0].set_ylim(0, 1)
    axs0_right.set_ylabel('RMSE, MAE')

    # 双y轴legend
    lines = a + b + c
    axs[0].legend(handles=lines, labels=[l.get_label() for l in lines], loc='best', frameon=False, ncols=3)
    
    # 网格线
    axs[0].grid(which='major', axis='y', color='silver', lw=1.0, ls='-')

    # 作图：slope
    axs[1].plot(list_slope, marker='o', label='slope', color='black')
    axs[1].set_ylabel('Slope')

    # 作图：残差
    axs[2].boxplot(
        list_residual_err, 
        positions=range(len(valid_names)),
        widths=0.8,
        showfliers=False, showmeans=True,  # 不显示outlier, 显示平均值点
        showcaps=False,  # 不显示whisker两端水平线
        whis=(5, 95),  # 设置whisker范围为5-95%
        boxprops=dict(linewidth=1.5, color='black'),  # box框框格式
        meanprops=dict(marker='s', markersize=markersize_mean, markerfacecolor='limegreen', markeredgecolor='blue', markeredgewidth=1.5),  # 均值点格式
        medianprops=dict(linestyle='-', linewidth=markersize_median, color='red'),  # 中位数线格式
        whiskerprops=dict(linewidth=1.5)  # whisker格式
    )
    axs[2].set_ylabel('Residual Error')

    # 作图：cv-r2
    axs[3].boxplot(
        list_cv_r2, 
        positions=range(len(valid_names)),
        widths=0.8,
        showfliers=False, showmeans=True,  # 不显示outlier, 显示平均值点
        showcaps=False,  # 不显示whisker两端水平线
        whis=(5, 95),  # 设置whisker范围为5-95%
        boxprops=dict(linewidth=1.5, color='black'),  # box框框格式
        meanprops=dict(marker='s', markersize=markersize_mean, markerfacecolor='limegreen', markeredgecolor='blue', markeredgewidth=1.5),  # 均值点格式
        medianprops=dict(linestyle='-', linewidth=markersize_median, color='red'),  # 中位数线格式
        whiskerprops=dict(linewidth=1.5)  # whisker格式
    )
    axs[3].set_ylabel('CV-R$^2$')

    axs[3].set_xlim(-0.5, len(valid_names) - 0.5)
    axs[3].set_ylim(0, 1)

    base.set_locator(ax=axs[0], which='y', ylocator=mticker.MultipleLocator(0.2))
    base.set_locator(ax=axs0_right, which='y')
    base.set_locator(ax=axs[1], which='y')
    base.set_locator(ax=axs[2], which='y')
    base.set_locator(ax=axs[3], which='y', ylocator=mticker.MultipleLocator(0.2))

    # 网格线
    axs[3].grid(which='major', axis='y', color='silver', lw=1.0, ls='-')

    # 如果ncols=1，则隐藏x轴刻度
    if ncols == 1:

        # 隐藏刻度
        for ax in axs:
            ax.tick_params(axis='x', which='both', length=0)
        
        # 隐藏ticklabels
        axs[3].set_xticklabels([])
    
    elif ncols == 2:

        for i in [2, 3]:
                
            # 设置xtick
            axs[i].set_xticks(range(len(valid_names)))
            
            # 设置xticklabels
            axs[i].set_xticklabels(valid_names, rotation=rotation_x, ha='center')
    
    elif ncols == 4:

        for i in range(4):
            
            # 设置xtick
            axs[i].set_xticks(range(len(valid_names)))
            
            # 设置xticklabels
            axs[i].set_xticklabels(valid_names, rotation=rotation_x, ha='center')

    plt.show()


if __name__ == '__main__':

    dir_project = Path(r'F:\_timingBackup_src\02成果\Paper08@2024_OptNOx\data\model01-site-daily')

    # h5文件目录
    dir_hdf5 = dir_project / 'h5'

    list_site = [i.stem[:5] for i in dir_hdf5.glob('*.h5') if i.is_file()]
    print(list_site)
    # exit(0)

    plot_summary(
        dir_project=dir_project,
        name_prefix=list_site,
        figsize=(25, 12),
        ncols=1,
        markersize_mean=3,
        markersize_median=2,
    )
