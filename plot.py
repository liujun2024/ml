""" 机器学习作图相关的函数 """

from __future__ import annotations
# import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.cm as cm
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn as sns
# from matplotlib import rcParams, colors, ticker
# from ml_main import cal_yearly_seasonal, cal_bin_mean_equidistance
# import config as cfg
# import ml_fitting as fit

# from . import _hdf5 as hdf5
from ml import hdf5
# import _hdf5 as hdf5
# from allin1.ml import HDF5RW


# 作图默认参数控制
plt.rcParams["figure.dpi"] = 100  # 图片显示默认分辨率
plt.rcParams['savefig.dpi'] = 300  # 图片保存默认分辨率
plt.rcParams['savefig.transparent'] = True  # 图片保存透明背景
plt.rcParams['font.size'] = 16  # 图片默认字号
plt.rcParams['axes.unicode_minus'] = False  # 作图时正常显示符号
plt.rcParams['axes.linewidth'] = 1.2  # spine 边框线宽
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'  # 作图字体：微软雅黑（同时支持中英文）
# plt.rcParams['font.sans-serif'] = 'Times New Roman'  # 作图字体：微软雅黑（同时支持中英文）
for v in ['xtick', 'ytick']:
    plt.rcParams[v + '.major.size'] = 6  # 主刻度线长
    plt.rcParams[v + '.minor.size'] = 4  # 次刻度线长
    plt.rcParams[v + '.major.width'] = 1.2  # 主刻度线宽
    plt.rcParams[v + '.minor.width'] = 1.2  # 次刻度线宽


def performance_ts(data_raw: pd.Series, data_predict: pd.Series, annotation_: str, ax: Axes):
    """ 模型性能作图中的时间序列 
        
        data_raw: pd.Serise, 观测数据, index为pd.DatetimeIndex
        data_predict: pd.Serise, 预测数据, index为pd.DatetimeIndex
        type_: str, 注释, "Train" 或 "Test"
    
        无返回值
    2023-12-27 v1
    单进程
    """

    # 时间序列-观测值
    data_raw.plot.line(color='grey', ax=ax, zorder=0, label='Observation', lw=0.5)
    
    # 时间序列-预测值
    data_predict.plot.line(color='black', ax=ax, zorder=1, label='Prediction', lw=0.5, alpha=1)

    # 标明train/test
    ax.text(x=0.02, y=0.95, s=annotation_, color='black', ha='left', va='top', transform=ax.transAxes, fontsize=20)

    # xlabel、ylabel
    ax.set_ylabel(data_raw.name)

    # 图例
    ax.legend(loc='upper right', frameon=False, ncol=2)


def performance_ts_v2(data_:dict, annotation_: str, ax: Axes):
    """ 模型性能作图中的时间序列 

        data_: pd.DataFrame, index为pd.DatetimeIndex, 第1,2列分别为原始值与预测值
        annotation_: str, 注释, "Train" 或 "Test"
    
        无返回值
    2023-12-27 v1
    单进程
    """

    # 提取数据
    data_raw = data_.iloc[:, 0]
    data_predict = data_.iloc[:, 1]

    # 时间序列-观测值
    data_raw.plot.line(color='black', ax=ax, zorder=0, label='Observation', lw=0.5)
    
    # 时间序列-预测值
    data_predict.plot.line(color='red', ax=ax, zorder=1, label='Prediction', lw=0.5, alpha=1)

    # 标明train/test
    ax.text(x=0.02, y=0.95, s=annotation_, color='black', ha='left', va='top', transform=ax.transAxes, fontsize=20)

    # xlabel、ylabel
    ax.set_xlabel('')
    ax.set_ylabel(data_raw.name)

    # 图例
    ax.legend(loc='upper right', frameon=False, ncol=2)


def performance_scatter(data_:dict, annotation_: str, ax: Axes):
    """ 模型性能作图中的散点图和直方图
        
        data_: dict,
            {
                'r2': float, 
                'rmse': float,
                'df': pd.DataFrame, index为pd.DatetimeIndex, 第1,2列分别为原始值与预测值
            }
        
        annotation_: str, 注释, "Train" 或 "Test"
    
        无返回值
    2023-12-27 v1
    单进程
    """
    # print(data_)
    
    # 提取数据
    array1d_raw = data_['df'].iloc[:, 0].to_numpy()
    array1d_predict = data_['df'].iloc[:, 1].to_numpy()
    # print(array1d_raw)
    # print(array1d_predict)

    # 原始数据散点图
    ax.scatter(x=array1d_raw, y=array1d_predict,
               s=50, marker="$\u25EF$", alpha=0.8, color='red', lw=0.1, zorder=0)
            #    s=50, marker="$\u25EF$", alpha=0.8, color='darkgray', lw=0.1, zorder=0)
    
    # 线性拟合: slope, intercept, r, p, stderr_slope, stderr_intercept
    fitting_result_train = stats.linregress(x=array1d_raw, y=array1d_predict)

    # 斜率和截距
    slope_train, intercept_train = fitting_result_train[:2]

    # 1:1线端点
    y_11_train = (0, array1d_raw.max())

    # 1:1线
    ax.plot((0, array1d_raw.max()), y_11_train, color='black', lw=2, label='1:1', ls='--', zorder=2)

    # 拟合线端点
    fitting_y_train = (slope_train * array1d_raw.min() + intercept_train, 
                       slope_train * array1d_raw.max() + intercept_train)

    # 拟合线
    ax.plot((array1d_raw.min(), array1d_raw.max()), fitting_y_train, color='green', lw=2, label='linear fitting', zorder=3)
    # ax.plot((array1d_raw.min(), array1d_raw.max()), fitting_y_train, color='red', lw=2, label='linear fitting', zorder=3)

    # 注释内容
    annotation_train = "${y=%.2fx + %.2f}$\n${R^2=%.2f}$\n${RMSE=%.2f}$\n${MAE=%.2f}$" % (slope_train, intercept_train, data_['r2'], data_['rmse'], data_['mae'])

    # 注释
    ax.text(x=0.05, y=0.82, s=annotation_train, color='green', ha='left', va='top', transform=ax.transAxes)
    # ax.text(x=0.05, y=0.95, s=annotation_train, color='red', ha='left', va='top', transform=ax.transAxes)

    # 统计直方图-x轴
    ax_hist1 = ax.twinx()
    ax_hist1.hist(x=array1d_raw, bins=50, histtype='bar', color='silver', edgecolor='gray', lw=0.1, alpha=0.5, zorder=5)
    # ax_hist1.hist(x=array1d_raw, bins=50, histtype='step', color='black', edgecolor='#9467bd', lw=1, alpha=1, zorder=5)
    # ax_hist1.hist(x=array1d_raw, bins=50, histtype='bar', color='orange', edgecolor='darkorange', lw=0.1, alpha=0.3, zorder=5)

    # 统计直方图-y轴
    # plt.hist()
    ax_hist2 = ax.twiny()
    ax_hist2.hist(x=array1d_predict, bins=50,histtype='bar', color='silver', edgecolor='gray', lw=0.1, alpha=0.5, zorder=5, orientation='horizontal')
    # ax_hist2.hist(x=array1d_predict, bins=50, histtype='step', color='black', edgecolor='#17becf', lw=1, alpha=1, zorder=5, orientation='horizontal')
    # ax_hist2.hist(x=array1d_predict, bins=50,histtype='bar', color='orange', edgecolor='darkorange', lw=0.1, alpha=0.3, zorder=5, orientation='horizontal')

    # 直方图y轴范围降到1/10
    ax_hist1.set_ylim(0, ax_hist1.get_ylim()[1] * 10)
    ax_hist2.set_xlim(0, ax_hist2.get_xlim()[1] * 10)

    # 直方图关闭坐标轴
    # ax_y1.tick_params(labelright=False, right=False)
    ax_hist1.set_axis_off()
    ax_hist2.set_axis_off()

    # 标明train/test
    ax.text(x=0.05, y=0.95, s=f'{annotation_} (N={array1d_raw.shape[0]})', color='black', ha='left', va='top', transform=ax.transAxes, fontsize=20)
    # ax.text(x=0.6, y=0.95, s=annotation_, color='black', ha='left', va='top', transform=ax.transAxes, fontsize=20)

    # xlabel、ylabel
    ax.set_xlabel('Observation')
    ax.set_ylabel('Prediction')

    # 图例
    ax.legend(loc='lower right', frameon=False, handlelength=2.3)


def shap_dependence(data_shap: pd.DataFrame, data_raw: pd.DataFrame, path_png: str | Path | None = None, dpi : int = 100):
    """ shapley values 依赖图 

    Parameters
    ----------
    data_shap : pd.DataFrame
        shapley values，含有datetime索引，所有列均为自变量
    data_raw : pd.DataFrame
        训练数据（观测数据），含有和data_shap完全相同的datetime索引，自变量，最后一列为因变量
    path_png : str, optional
        图片保存路径

    2025.06.05  Created by LiuJun
    """

    # 检查data_shap与data_raw列名顺序是否一致
    if data_shap.columns.tolist() != data_raw.columns.tolist()[:-1]:
        raise ValueError('data_shap和data_raw列名不一致！')

    # 作图行列数
    plot_rows = math.floor(data_shap.shape[1] ** 0.5)
    plot_cols = math.ceil(data_shap.shape[1] / plot_rows)

    # 画布设置
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(18, 10), layout='constrained')    # type: ignore
    if plot_rows * plot_cols == 1:
        ax: list[Axes] = [ax]   # type: ignore
    else:
        ax: list[Axes] = ax.flatten()   # type: ignore

    # 计算全局shap值
    df_shap_global = data_shap.abs().mean(axis=0)

    # 降序排列
    df_shap_global.sort_values(ascending=False, inplace=True)

    # n = 0
    # for m in data_shap.columns:
    for n, m in enumerate(df_shap_global.index):

        # 散点map图
        scatter_n = ax[n].scatter(
            x=data_raw.loc[:, m],  # x
            y=data_shap.loc[:, m],  # y
            s=15,  # 大小
            c=data_raw.iloc[:, -1],  # 颜色
            marker=".",  # 点
            # marker="$\u25EF$",  # 空心圆圈
            alpha=0.8,  # 透明度
            cmap='jet',  # 颜色映射
            lw=0.25,  # 线宽
            norm=mcolors.LogNorm(),  # cmap对数
        )

        # 统计直方图
        ax_in = ax[n].inset_axes(bounds=(0, 1.0, 1, 0.15), sharex=ax[n])
        ax_in.hist(x=data_raw.loc[:, m], bins=50, histtype='bar', color='silver', edgecolor='grey', lw=0.1)

        # 直方图关闭坐标轴，只保留数据
        ax_in.set_axis_off()

        # 轴标签
        # ax[n].set_xlabel(suptitle)
        ax[n].set_ylabel('shap')

        # colorbar
        cb = fig.colorbar(scatter_n, ax=ax[n], extend='neither')
        # cb = fig.colorbar(scatter_n, ax=ax[n], extend='both')

        # colorbar标题
        cb.ax.set_title(data_raw.columns[-1], fontsize=10)

        # 子图标题
        ax[n].set_title(m)

        # n += 1

    # 关闭多余的子图
    for i in range(data_shap.shape[1], plot_rows * plot_cols):
        ax[i].set_axis_off()

    # 图像标题
    # plt.suptitle(suptitle, x=0.5, y=0.99)
    # ax[0].set_title(suptitle, x=0.5, y=1.01)

    # 窗口标题
    # fig.canvas.manager.set_window_title(suptitle)

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.95)

    if path_png:
        plt.savefig(path_png, dpi=dpi)
        plt.close()
    else:
        plt.show()


def shap_dependence_base(data_plot: pd.DataFrame, ax: Axes, title_cb: str = '', ylabel_shap: str = 'SHAP value (μg·m$^{-3}$)'):
    """ 绘制单个特征的dependence图 
    
    Parameters
    ----------
    data_plot : pd.DataFrame
        含有datetime索引，x为自变量，y为因变量值，shap为shapley values

    """

    # 散点图
    scatter = ax.scatter(
            x=data_plot.loc[:, 'x'],  # x
            y=data_plot.loc[:, 'shap'],  # y
            s=15,  # 大小
            c=data_plot.loc[:, 'y'],  # 颜色
            # marker=".",  # 点
            marker="$\u25EF$",  # 空心圆圈
            alpha=0.8,  # 透明度
            cmap='YlGn',  # 颜色映射
            lw=0.25,  # 线宽
            # norm=mcolors.LogNorm(),  # cmap对数
        )

    # 统计直方图
    ax_in = ax.inset_axes(bounds=(0, 1.0, 1, 0.10), sharex=ax)
    ax_in.hist(x=data_plot.loc[:, 'x'], bins=50, histtype='bar', color='silver', edgecolor='grey', lw=0.1)

    # 直方图关闭坐标轴，只保留数据
    ax_in.set_axis_off()

    # 轴标签
    # ax[n].set_xlabel(suptitle)
    ax.set_ylabel(ylabel_shap)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cb = plt.colorbar(scatter, cax=cax, extend='neither')
    # cb = fig.colorbar(scatter, ax=ax, extend='neither')
    # cb = fig.colorbar(scatter_n, ax=ax[n], extend='both')

    # colorbar标题
    cb.ax.set_title(title_cb, fontsize='small')

    # 返回ax
    return ax


def plotShapDependence(data_shap: pd.DataFrame, data_raw: pd.DataFrame, y: str, x: list = [], path_png: str | bool = False, suptitle=''):
    """ 浓度shapley values对应图（按自变量分类）

        data_shap: pd.DataFrame，Shapley values，含有datetime索引，所有列均为自变量
        data_train_x: pd.DataFrame, 训练数据（观测数据），含有和data_shap完全相同的datetime索引，自变量
        data_train_y: pd.DataFrame, 训练数据（观测数据），含有和data_shap完全相同的datetime索引，因变量
        path_png: str, 图片保存路径
        y: 因变量名
        suptitle: str, 主标题

    无返回值
    2023-06-25 v1
    单进程
    """

    if len(x) != 0:

        # 原始数据
        data_raw = data_raw.loc[:, x + [y]]

        # shap数据
        data_shap = data_shap.loc[:, x]

    # 特征列表
    list_name = data_shap.columns.to_list()

    # 作图行列数
    plot_rows = math.floor(len(list_name) ** 0.5)
    plot_cols = math.ceil(len(list_name) / plot_rows)

    # 画布设置
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(18, 10), layout='constrained')
    if plot_rows * plot_cols == 1:
        ax = [ax]
    else:
        ax = ax.flatten()

    n = 0
    for m in list_name:
        # 散点map图
        scatter_n = ax[n].scatter(
            x=data_raw.loc[:, m],  # x
            y=data_shap.loc[:, m],  # y
            s=15,  # 大小
            c=data_raw.loc[:, y],  # 颜色
            marker=".",  # 点
            # marker="$\u25EF$",  # 空心圆圈
            alpha=0.8,  # 透明度
            cmap='jet',  # 颜色映射
            lw=0.25,  # 线宽
            norm=mcolors.LogNorm(),  # cmap对数
        )

        # 统计直方图
        ax_in = ax[n].inset_axes(bounds=(0, 1.0, 1, 0.15), sharex=ax[n])
        ax_in.hist(x=data_raw.loc[:, m], bins=50, histtype='bar', color='silver', edgecolor='grey', lw=0.1)

        # 直方图关闭坐标轴，只保留数据
        ax_in.set_axis_off()

        # 轴标签
        ax[n].set_xlabel(suptitle)
        ax[n].set_ylabel('shap')

        # colorbar
        cb = fig.colorbar(scatter_n, ax=ax[n], extend='neither')
        # cb = fig.colorbar(scatter_n, ax=ax[n], extend='both')

        # colorbar标题
        cb.ax.set_title(y, fontsize=10)

        # 子图标题
        ax[n].set_title(m)

        n += 1

    # 关闭多余的子图
    for i in range(len(list_name), plot_rows * plot_cols):
        ax[i].set_axis_off()

    # 图像标题
    plt.suptitle(suptitle, x=0.5, y=0.99)
    # ax[0].set_title(suptitle, x=0.5, y=1.01)

    # 窗口标题
    fig.canvas.manager.set_window_title(suptitle)

    # plt.tight_layout()
    # plt.subplots_adjust(top=0.95)

    if path_png:
        plt.savefig(path_png, transparent=True, dpi=300)
        plt.close()
    else:
        plt.show()


# def plotRMatrix(path_h5: Path, figsize: tuple = (12, 8), path_png: Path | None = None, dpi: int = 100):
def plotRMatrix(
        data_raw: pd.DataFrame, 
        # path_h5: Path, 
        figsize: tuple = (18, 12), 
        dict_rename: dict = {},
        path_png: Path | None = None, 
        dpi: int = 100,
        show : bool = False,
    ):
    
    """ Pearson相关系数（R）矩阵作图 
    
    Parameters
    ----------
    data_raw : pd.DataFrame
        训练数据（观测数据），含有和data_shap完全相同的datetime索引，自变量，最后一列为因变量

    figsize : tuple, optional
        画布大小, by default (10, 10)
    
    dict_rename : dict, optional
        列名映射， by default {}

    path_png : Path, optional
        保存路径, by default None
    
    dpi : int, optional
        分辨率, by default 100
    
    """

    # 读取数据
    # df = hdf5.HDF5RW(path_h5=path_h5).df_raw
    data_raw.to_csv(path_png.with_suffix('.csv'))

    # 计算相关系数矩阵
    df_r = data_raw.corr(method='pearson')
    # df_r = data_raw.corr(method='pearson')[::-1]

    # 只保留左下三角矩阵
    arr2d_r = np.tril(df_r.to_numpy(), -1)[1:, :-1]

    # 上三角元素设置为nan
    arr2d_r[np.triu_indices_from(arr2d_r, 1)] = np.nan

    # print(data_y.shape)
    label_x = df_r.columns[:-1]  # 表头，横轴
    label_y = df_r.index[1:]  # 纵轴，索引
    # print(label_x, label_y)

    # 列名映射
    if len(dict_rename) != 0:
        label_x = [dict_rename.get(i, i) for i in label_x]
        label_y = [dict_rename.get(i, i) for i in label_y]

    # 对角线1替换为nan
    # array_r = df_r.to_numpy()
    # array_r[np.where(array_r == 1)] = np.nan

    # 画布大小
    # if figsize is None:
        # figsize = (12, 8)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100, layout='constrained')
    # fig.canvas.manager.set_window_title("Pearson's R matrix")  # 窗口标题
    heatmap = ax.imshow(arr2d_r, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    # heatmap = ax.imshow(data_y, cmap='jet', vmin=-1, vmax=1)

    # x,y轴刻度及标签
    ax.set_xticks(np.arange(len(label_x)), minor=False)
    ax.set_xticks(np.arange(len(label_x))-0.5, minor=True)
    ax.set_yticks(np.arange(len(label_y)), minor=False)
    ax.set_yticks(np.arange(len(label_y))-0.5, minor=True)
    ax.set_xticklabels(label_x)
    ax.set_yticklabels(label_y)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")  # x轴标签旋转45°

    # ax.set_xticks(range(len(label_x)))
    # ax.set_yticks(range(len(label_y)))
    # ax.set_xticklabels(label_x)
    # ax.set_yticklabels(label_y)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")  # x轴标签旋转45°

    # 数值标注的字体大小
    if len(label_y) < 5:
        fs = 'large'
    elif len(label_y) < 10:
        fs = 'medium'
    elif len(label_y) < 15:
        fs = 'small'
    elif len(label_y) < 20:
        fs = 'x-small'
    else:
        fs = 'xx-small'

    # 添加数值标注
    valfmt = mticker.StrMethodFormatter('{x:.2f}')  # 标注保留两位小数
    for i in range(arr2d_r.shape[0]):
        for j in range(arr2d_r.shape[1]):
            if not np.isnan(arr2d_r[i, j]):
                heatmap.axes.text(j, i, valfmt(arr2d_r[i, j]), ha='center', va='center', color='black', fontsize=fs)

    cbar = plt.colorbar(heatmap, pad=0.01)  # 添加colorbar
    cbar.set_ticks(np.arange(-1, 1.05, 0.2))

    # 去除图片边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 设置网格线
    ax.grid(which='minor', axis='both', linestyle='-', color='white', alpha=1, linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # 显示图片
    if show:
        plt.show()
    
    # 保存图片
    if path_png:
        plt.savefig(path_png, dpi=dpi, transparent=False)
        plt.close()
    

def plotRankingSHAP(
        data: list[dict], 
        figsize: tuple = (8, 6),
        nrows: int = None,
        path_png: Path = None, 
        dpi: int = 100,
        ):
    
    """ 特征重要性排序图 
    Parameters
    ----------
    data : list[dict]， dict：{'Series': pd.Series特征重要性排序Global SHAP, 'title': 子图标题}
    figsize : tuple， 画布比例
    nrows : int， 子图行数
    path_png : Path， 图片保存路径
    dpi : int， 分辨率，图片保存分辨率
    """

    # 作图行列数
    if nrows is None:
        nrows = round(len(data) ** 0.5)  # 作图行数
    ncols = math.ceil(len(data) / nrows)  # 作图列数
    
    # 画布设置
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=100,
        sharex=False,
        sharey=False,
        layout='constrained',
    )

    if len(data) == 1:
        ax : list[plt.Axes] = [ax]
    else:
        ax : list[plt.Axes] = ax.flatten()

    # 遍历子图
    for i, d in enumerate(data):
        
        # Series
        s = d['Series']

        # 排序
        # pd.Series().sort_values()
        s.sort_values(ascending=True, inplace=True)

        # 计算百分比
        s_percent = s / s.sum() * 100

        # 作图
        ax_container = ax[i].barh(y=s.index, width=s.values, color='#f42756', height=0.75)

        # xlabel
        ax[i].set_xlabel(r'$\overline{\mathrm{|shap|}}$')

        # 子图标题
        if 'title' in d.keys():
            ax[i].set_title(d['title'])

        # 设置label
        labels = ['%.1f%%' % v for v in s_percent]
        ax[i].bar_label(container=ax_container, labels=labels, fmt='%.1f', label_type='edge')

        # xlim、ylim
        ax[i].set_xlim((0, ax[i].get_xlim()[1] * 1.15))
        ax[i].set_ylim((-0.75, s.shape[0] - 0.25))

    # 保存图片
    if path_png:
        plt.savefig(path_png, dpi=dpi)
    
    plt.show()


def plot_beeswarm(
        raw: pd.DataFrame, 
        shap: pd.DataFrame, 
        ax1: Axes,
        ax2: Axes,
        path_png: Path | None = None, 
        dpi=100, 
        head: int = 10,
        figsize: tuple = (8, 6),
    ):
    """ 柱状图+蜂群图 """

    from ._plot_beeswarm import summary_legacy
    from ._colors import red_blue

    # 计算绝对值均值
    df_shap_global = shap.abs().mean(axis=0).sort_values(ascending=True).tail(head)
    print('df_shap_global:\n', df_shap_global)

    # 准备画布
    # fig, (ax1, ax2) = plt.subplots(figsize=figsize, ncols=2, nrows=1, dpi=100, sharey=False)

    # 特征重要性柱状图
    df_shap_global.plot.barh(width=0.7, color='#1e87e4', ax=ax1, zorder=10)

    # 网格线
    ax1.grid(visible=True, which='major', axis='y', color="#cccccc", lw=0.5, dashes=(1, 4), zorder=0)

    # 获取当前的 Axes 对象，并将 scatter 图形对象添加到子图中
    ax_ = summary_legacy(shap.to_numpy(), raw.to_numpy(), feature_names=shap.columns.tolist())

    scatter_path_collection = ax_.collections[0]

    # 将散点图的PathCollection对象添加到子图中
    ax2.add_collection(scatter_path_collection)

    ax2.set_xlabel('SHAP value', verticalalignment='center_baseline', fontsize=18)

    # # 全国统一模型前10
    # ax[5].barh(['SO2', 'CO', 'NO2', '农田', 'O3'][::-1], [4.69, 2.37, 1.56, 1.48, 1.47][::-1], height=0.6)
    # ax[5].set_title('全国统一模型', fontsize=20)
    ax1.set_xlabel('mean(|SHAP value|)', verticalalignment='center_baseline', fontsize=18)
    # ax.set_xlabel('$\overline{\mathrm{|SHAP\;value|}}$', verticalalignment='center_baseline', fontsize=12)

    # ax[5].set_axis_off()

    ax2.set_yticklabels([])

    ax1.set_title('Importance plot')
    ax2.set_title('Summary plot')

    ax2.set_ylim(ax1.get_ylim())

    y_ticklabel = ax1.get_yticklabels()
    ax1.set_yticklabels(y_ticklabel, fontsize=18)

    x1_ticklabel = ax1.get_xticklabels()
    ax1.set_xticklabels(x1_ticklabel, fontsize=18)

    x2_ticklabel = ax2.get_xticklabels()
    ax2.set_xticklabels(x2_ticklabel, fontsize=18)

    # colorbar
    # ax_cbar = ax1.inset_axes([1, 0, 0.2, 1])

    ax_cbar = ax2.inset_axes(bounds=(1.02, 0, 0.02, 1))

    m = cm.ScalarMappable(cmap=red_blue)
    m.set_array([0, 1])
    cb = plt.colorbar(m, cax=ax_cbar, extend='neither', ticks=[0.015, 0.985])
    cb.set_ticklabels(['low', 'high'])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=16, length=0)

    # colorbar标题
    # cb_d.ax.set_title(data_.columns[-1], fontsize=10)

    # cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
    # # cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
    cb.set_label('Feature value', size='medium', labelpad=0, va='bottom')
    # cb.set_alpha(1)

    # 子图编号
    # ax1.text(0.9, 0.16, '(c)', fontsize='x-large', ha='center', va='top', transform=ax1.transAxes)
    # ax2.text(0.9, 0.16, '(d)', fontsize='x-large', ha='center', va='top', transform=ax2.transAxes)

    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1)
    # plt.show()


def plotShapRanking(
        raw: pd.DataFrame,
        shap: pd.DataFrame,
        list_x: list[str],
        path_png: Path | None = None, 
        dpi=100, 
        head: int | None = None,
        figsize: tuple = (8, 10),
        show: bool = False,
    ):
    """ 基于shap值的特征重要性排序，含beeswarm图

    Parameters
    ----------
    shap : pd.DataFrame， shap值
    path_png : Path | None， 图片保存路径
    dpi : int， 分辨率，图片保存分辨率
    head : int | None， 重要性排序的前几项
    figsize : tuple， 画布大小
    
    2025.08.14 合并相关函数
    """

    from ._plot_beeswarm import summary_legacy
    from ._colors import red_blue

    # 复制可变对象，避免修改原始数据
    df_raw = raw.copy()
    df_shap = shap.copy()

    # 计算全局shap值
    df_shap_global = df_shap.abs().mean(axis=0)

    # 升序排列
    df_shap_global.sort_values(ascending=True, inplace=True)

    # 计算百分比
    df_shap_global_percent = df_shap_global / df_shap_global.sum() * 100

    # 截取最大的head个特征
    if head is not None:
        df_shap_global = df_shap_global.tail(head)
        df_shap_global_percent = df_shap_global_percent.tail(head)

    # 画布设置
    _, axs = plt.subplots(figsize=figsize, ncols=2, nrows=1, dpi=dpi, sharey=False, layout='none')

    axs : list[Axes] = axs.flatten()    # type: ignore

    # 特征重要性
    df_shap_global.plot.barh(width=0.7, color='#1e87e4', ax=axs[0], zorder=10)
    
    # 添加百分比标签
    for i, v in enumerate(df_shap_global_percent):
        axs[0].text(
            x=axs[0].get_xlim()[1] * 0.01, y=i*0.997, s=f'{v:.1f}%', 
            va='center', fontsize='medium', zorder=20, color='black', 
        )

    # 网格线
    axs[0].grid(visible=True, which='major', axis='y', color="#cccccc", lw=0.5, dashes=(4, 2), zorder=-1)
    # ax1.grid(visible=True, which='major', axis='y', color="#cccccc", lw=0.5, dashes=(1, 4), zorder=0)

    # 做beeswarm图，并获得Axes对象
    ax_ = summary_legacy(
        shap_values=df_shap.to_numpy(), 
        features=df_raw.loc[:, list_x].to_numpy(),      # type: ignore
        feature_names=df_shap.columns.tolist(),
        max_display=df_shap.shape[1] if head is None else head,
    )

    # 将beeswarm图的PathCollection对象添加到子图ax2中
    axs[1].add_collection(ax_.collections[0])

    # x轴标题设置
    axs[0].set_xlabel('mean(|SHAP value|)', verticalalignment='center_baseline')
    axs[1].set_xlabel('SHAP value', verticalalignment='center_baseline')

    # 删除ax2的y轴刻度标签
    axs[1].set_yticklabels([])

    # 对齐ax1和ax2的y轴范围
    axs[1].set_ylim(axs[0].get_ylim())

    # ax2图的colorbar
    ax_cbar = axs[1].inset_axes(bounds=(1.02, 0, 0.03, 1))

    # 使用预设的colormap
    m = cm.ScalarMappable(cmap=red_blue)

    # 设置刻度
    m.set_array([0, 1])

    # 绘制colorbar
    cb = plt.colorbar(m, cax=ax_cbar, extend='neither', ticks=[0.01, 0.99])

    # 刻度值
    cb.set_ticklabels(['low', 'high'])
    
    # 隐藏边框
    cb.outline.set_visible(False)   # type: ignore
    
    # 刻度值大小
    cb.ax.tick_params(labelsize=14, length=0)

    # colorbar标题
    # cb.set_label('Feature value', size=14, labelpad=-12, va='bottom')
    cb.set_label('Feature value', labelpad=-12, va='bottom')
    # cb.set_alpha(1)

    # 调整子图间距为0
    # fig.set_constrained_layout_pads(hspace=0, wspace=0)   # type: ignore
    # fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)   # type: ignore
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(wspace=0)

    # 保存图片
    if path_png is not None:
        plt.savefig(path_png, dpi=100)
    
    # 显示图片
    if show:
        plt.show()
    else:
        plt.close()


# def beeswarm_base(
#         data_raw: pd.DataFrame, 
#         data_shap: pd.DataFrame, 
#         ax: Axes,
#         show_colorbar: bool = True,
#         show: bool = False,
#     ):
#     """ 
#     仅绘制蜂群beeswarm图

#     Parameters
#     ----------
#     data_raw : pd.DataFrame， 原始数据
#     data_shap : pd.DataFrame， shap值
#     ax : Axes， 画布对象
#     # head : int， 绘制重要性排序的前几个特征
    
#     2025-11-19  v1  从plot_beeswarm中提取beeswarm图绘制代码，并修改为仅绘制beeswarm图
#     """
    
#     # from shap.plots._beeswarm import summary_legacy
#     # from shap.plots.colors import red_blue

#     # from ._plot_beeswarm import summary_legacy
#     # from ._colors import red_blue

#     print('11')

#     # 计算绝对值均值
#     # df_shap_global = data_shap.abs().mean(axis=0).sort_values(ascending=True).tail(head)

#     # 前head个特征列表
#     # list_head = df_shap_global.index.tolist()

#     # print('df_shap_global:\n', df_shap_global)

#     # 准备画布
#     # fig, (ax1, ax2) = plt.subplots(figsize=figsize, ncols=2, nrows=1, dpi=100, sharey=False)

#     # 特征重要性柱状图
#     # df_shap_global.plot.barh(width=0.7, color='#1e87e4', ax=ax1, zorder=10)

#     # 网格线
#     # ax1.grid(visible=True, which='major', axis='y', color="#cccccc", lw=0.5, dashes=(1, 4), zorder=0)

#     # 获取当前的 Axes 对象，并将 scatter 图形对象添加到子图中
#     summary_legacy(
#         df_shap=data_shap, 
#         df_raw=data_raw, 
#         # feature_names=list_head,
#         # plot_type='dot',
#         # show=False,
#         ax=ax,
#         show_colorbar=show_colorbar,
#     )

#     # scatter_path_collection = ax_.collections[0]

#     # 将散点图的PathCollection对象添加到子图中
#     # ax.add_collection(scatter_path_collection)

#     ax.set_xlabel('SHAP value', verticalalignment='center_baseline', fontsize=18)

#     # # 全国统一模型前10
#     # ax[5].barh(['SO2', 'CO', 'NO2', '农田', 'O3'][::-1], [4.69, 2.37, 1.56, 1.48, 1.47][::-1], height=0.6)
#     # ax[5].set_title('全国统一模型', fontsize=20)
#     # ax1.set_xlabel('mean(|SHAP value|)', verticalalignment='center_baseline', fontsize=18)
#     # ax.set_xlabel('$\overline{\mathrm{|SHAP\;value|}}$', verticalalignment='center_baseline', fontsize=12)

#     # ax[5].set_axis_off()

#     # ax.set_yticklabels([])

#     # ax1.set_title('Importance plot')
#     # ax2.set_title('Summary plot')

#     # ax.set_ylim(ax1.get_ylim())

#     # y_ticklabel = ax1.get_yticklabels()
#     # ax1.set_yticklabels(y_ticklabel, fontsize=18)

#     # x1_ticklabel = ax1.get_xticklabels()
#     # ax1.set_xticklabels(x1_ticklabel, fontsize=18)

#     # x2_ticklabel = ax.get_xticklabels()
#     # ax.set_xticklabels(x2_ticklabel, fontsize=18)

#     # colorbar
#     # ax_cbar = ax1.inset_axes([1, 0, 0.2, 1])

#     # ax_cbar = ax.inset_axes(bounds=(1.02, 0, 0.02, 1))

#     # m = cm.ScalarMappable(cmap=red_blue)
#     # m.set_array([0, 1])
#     # cb = plt.colorbar(m, cax=ax_cbar, extend='neither', ticks=[0.015, 0.985])
#     # cb.set_ticklabels(['low', 'high'])
#     # cb.outline.set_visible(False)
#     # cb.ax.tick_params(labelsize=16, length=0)

#     # # colorbar标题
#     # # cb_d.ax.set_title(data_.columns[-1], fontsize=10)

#     # # cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
#     # # # cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
#     # cb.set_label('Feature value', size='medium', labelpad=0, va='bottom')
#     # # cb.set_alpha(1)


def shap_interaction_summary(
        arr3d_shap_interaction: np.ndarray, 
        df_raw: pd.DataFrame, 
        list_x: list,
        dpi: int = 100,
        path_png: Path | None = None,
        show: bool = False,
        ):
    """
    绘制shap交互作用图
    
    Parameters
    ----------
    arr3d_shap_interaction : np.ndarray， shap交互作用数组(n_samples, n_features, n_features)
    df_raw : pd.DataFrame， 原始数据
    list_x : list， 作图顺序
    dpi : int， 图片分辨率
    path_png : Path | None， 图片保存路径
    show : bool， 是否显示图片

    2025-11-19  v1  Created by LiuJun
    """

    """ 依次提取特征i与其它个特征的交互作用 """
    dict_ij = {}
    for i in range(len(list_x)):
        dict_i = {}
        for j in range(len(list_x)):
            dict_i[list_x[j]] = arr3d_shap_interaction[:, i, j]

        # 字典转DataFrame
        df_i = pd.DataFrame(dict_i)

        # 设置索引
        df_i.index = df_raw.index

        # columns重新排序
        df_i = df_i.loc[:, list_x]

        # 保存到字典中
        dict_ij[list_x[i]] = df_i

    # 设置画布
    fig, axs = plt.subplots(figsize=(16, 9), dpi=dpi, nrows=1, ncols=len(list_x), layout='constrained')
    axs : list[Axes] = axs.flatten()    # type: ignore

    # 依次绘制每个特征与其它特征的交互作用
    for i, d in enumerate(list_x):

        beeswarm_base(
            df_raw=df_raw.loc[:, list_x],
            df_shap=dict_ij[d],
            ax=axs[i],
            xlabel='',
            show_colorbar=True if i == len(list_x) - 1 else False,
            show_yticklabels=True if i == 0 else False,
            title=d,
        )

    fig.supxlabel('SHAP interaction value')

    # 保存图片
    if path_png is not None:
        plt.savefig(path_png, dpi=dpi)
    
    if show:
        plt.show()



def beeswarm_base(
        df_shap: pd.DataFrame, 
        df_raw: pd.DataFrame, 
        ax: Axes,
        alpha: float = 1.0, 
        title: str = 'title',
        xlabel: str = 'xlabel',
        show_colorbar: bool = True,
        show_outline: bool = False,
        show_yticklabels: bool = True,
    ):

    """ 
    绘制beeswarm图，基于shap.plots._beeswarm中提取的summary_legacy

    Parameters
    ----------
    df_shap : pd.DataFrame， shap值
    df_raw : pd.DataFrame， 原始数据
    ax : Axes， 画布对象
    alpha : float， marker透明度
    title : str， 标题
    xlabel : str， x轴标签
    show_colorbar : bool， 是否显示colorbar
    show_outline : bool， 是否显示边框
    show_yticklabels : bool， 是否显示y轴刻度标签

    2025-11-19  v1  Created by LiuJun
    """

    from shap.plots.colors import red_blue

    # 断言df_shap和df_raw的shape相同
    assert df_shap.shape == df_raw.shape, "df_shap和df_raw的shape必须相同！"

    # 断言df_shap和df_raw的列名相同
    assert df_shap.columns.tolist() == df_raw.columns.tolist(), "df_shap和df_raw的列名必须相同！"

    # 逆序columns
    df_shap = df_shap[df_shap.columns[::-1]].copy()
    df_raw = df_raw[df_raw.columns[::-1]].copy()

    # x=0的垂线
    ax.axvline(x=0, color="silver", zorder=-1, ls='--', lw=1.2)

    # 绘制每个特征的beeswarm图
    for pos, i in enumerate(df_shap.columns):

        # 水平线
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(4, 2), zorder=-1)
        
        # 提取当前特征的shap值和原始值
        shaps = df_shap.loc[:, i]
        values = df_raw.loc[:, i]

        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        values = values.iloc[inds].to_numpy()
        shaps = shaps.iloc[inds].to_numpy()

        N = len(shaps)

        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)

        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        
        ys *= 0.9 * (0.4 / np.max(ys + 1))

        # 筛选特征值范围: 5th-95th
        vmin = np.nanpercentile(values, 5)
        vmax = np.nanpercentile(values, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(values, 1)
            vmax = np.nanpercentile(values, 99)
            if vmin == vmax:
                vmin = np.min(values)
                vmax = np.max(values)
        
        if vmin > vmax: # fixes rare numerical precision issues
            vmin = vmax

        # 绘制nan值的beeswarm图为灰色
        nan_mask = np.isnan(values)
        ax.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777",
                    s=16, alpha=alpha, linewidth=0.05,
                    zorder=3, rasterized=len(shaps) > 500,
                    marker='$\u25EF$',
                    )

        # plot the non-nan values colored by the trimmed feature value
        cvals = values[np.invert(nan_mask)].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
        cvals[cvals_imp > vmax] = vmax
        cvals[cvals_imp < vmin] = vmin
        
        # 绘制非nan值的beeswarm图
        ax.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                    cmap=red_blue, vmin=vmin, vmax=vmax, s=16,
                    c=cvals, alpha=alpha, linewidth=0.05,
                    zorder=3, rasterized=len(shaps) > 500,
                    marker='$\u25EF$',
                    )
        
    # 设置标题
    ax.set_title(title)

    # 设置x轴标签
    ax.set_xlabel(xlabel)

    # 设置y轴刻度标签
    if show_yticklabels:
        ax.set_yticks(ticks=range(len(df_shap.columns)), labels=df_shap.columns.tolist())
    else:
        ax.set_yticklabels([])

    # 隐藏边框
    if not show_outline:
        for i in ['top', 'right', 'left']:
            ax.spines[i].set_visible(False)
        
        # 隐藏y轴刻度
        ax.yaxis.set_ticks_position('none')

    # ax.tick_params('y', length=20, width=0.5, which='major')

    # y轴范围
    ax.set_ylim(-1, df_shap.shape[1])

    # 绘制colorbar
    if show_colorbar:

        ax_cbar = ax.inset_axes(bounds=(1.05, 0, 0.03, 1))
        m = cm.ScalarMappable(cmap=red_blue)
        m.set_array([0, 1])
        cb = plt.colorbar(m, cax=ax_cbar, extend='neither', ticks=[0.015, 0.985])
        cb.set_ticklabels(['low', 'high'])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=16, length=0)

        # colorbar标题
        cb.set_label('Normalized feature value', size='medium', labelpad=0, va='bottom')


if __name__ == '__main__':
    
    """ 性能曲线测试 """
    # h5文件路径
    # path_h5 = r'D:\Downloads\test\h5\test.h5'
    path_h5 = r'G:\_ING\Paper09@2024_O3,HaiNan√\_回复审稿人意见\不同模型性能对比\project\h5\Haikou_O3_region_mean_autumn&winter_hourly-without_transport.h5'
    
    # 读取数据
    data = hdf5.HDF5RW(path_h5=path_h5)
    data.read_performance()
    print(data.dict_model)
    # exit(0)

    # 画布设置
    fig, ax = plt.subplot_mosaic(
        mosaic=[
                ['a', 'a', 'b'],
                ['c', 'c', 'd'],
        ],
        # layout='constrained',
        # layout='tight',
        height_ratios=[1, 1],
        width_ratios=[4, 4, 4],
        figsize=(14, 8),
        # top=0.9,
    )

    # 训练数据及预测数据时间序列
    ax['a'].plot(data.index_train, data.y_train, color='grey', label='Observation', lw=0.5)
    ax['a'].plot(data.index_train, data.dict_model['rf']['predict_train'], color='black', label='Prediction', lw=0.5)

    # 测试数据及预测数据时间序列
    ax['c'].plot(data.index_test, data.y_test, color='grey', label='Observation', lw=0.5)
    ax['c'].plot(data.index_test, data.dict_model['rf']['predict_test'], color='black', label='Prediction', lw=0.5)

    # 标明train/test
    # ax['a'].text(x=0.02, y=0.95, s='Training', color='black', ha='left', va='top', transform=ax['a'].transAxes, fontsize=20)
    # ax['c'].text(x=0.02, y=0.95, s='Test', color='black', ha='left', va='top', transform=ax['c'].transAxes, fontsize=20)

    # xlabel、ylabel
    ax['a'].set_ylabel(data.list_y[0])
    ax['c'].set_ylabel(data.list_y[0])

    # 图例
    ax['a'].legend(loc='upper right', frameon=False, ncol=2)
    ax['c'].legend(loc='upper right', frameon=False, ncol=2)

    # df_predict_train
    df_predict_train = pd.DataFrame(
            data=np.array([data.y_train, data.dict_model['rf']['predict_train']]).T,
            columns=['obs', 'predict'],
            index=data.index_train,
            )
    
    df_predict_test = pd.DataFrame(
            data=np.array([data.y_test, data.dict_model['rf']['predict_test']]).T,
            columns=['obs', 'predict'],
            index=data.index_test,
            )

    # 散点图和直方图
    performance_scatter(
        data_={'r2': data.dict_model['rf']['r2_train'], 'rmse': data.dict_model['rf']['rmse_train'], 'df': df_predict_train},
        annotation_='Training',
        ax=ax['b'],
    )
    
    performance_scatter(
        data_={'r2': data.dict_model['rf']['r2_test'], 'rmse': data.dict_model['rf']['rmse_test'], 'df': df_predict_test},
        annotation_='Test',
        ax=ax['d'],
    )

    plt.tight_layout()

    # 显示图片
    plt.show()


