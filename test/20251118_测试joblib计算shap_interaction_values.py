import time
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt

# ========== 数据准备 ==========
print("=" * 60)
print("准备数据集")
print("=" * 60)


# ========== 计算函数 ==========
def cal_shap_interaction_values(explainer, data):
    """计算 SHAP 相互作用值"""
    return explainer.shap_interaction_values(data)


if __name__ == '__main__':

    # 生成回归数据集
    n_samples = 1000
    n_features = 20
    n_informative = 15

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=42,
        noise=10
    )

    print(f"原始数据形状:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")

    # 转换为 DataFrame
    X = pd.DataFrame(
        X, 
        columns=[f'Feature_{i}' for i in range(n_features)]
    )

    # ========== 关键修复：确保 X 和 y 对齐 ==========
    print(f"\n分割前检查:")
    print(f"  X 样本数: {len(X)}")
    print(f"  y 样本数: {len(y)}")
    print(f"  是否匹配: {len(X) == len(y)}")

    # 正确的 train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )

    print(f"\n分割后数据形状:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    # ========== 模型训练 ==========
    print("\n" + "=" * 60)
    print("训练随机森林回归模型")
    print("=" * 60)

    start = time.time()
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    # ✓ 正确：使用 y_train
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # ✓ 正确：使用 y_test
    r2_score = model.score(X_test, y_test)

    print(f"模型训练耗时: {train_time:.2f}s")
    print(f"模型 R² 分数: {r2_score:.4f}")

    # ========== Explainer 创建 ==========
    print("\n" + "=" * 60)
    print("创建 SHAP Explainer")
    print("=" * 60)

    start = time.time()
    explainer = shap.TreeExplainer(model)
    explainer_time = time.time() - start

    print(f"Explainer 创建耗时: {explainer_time:.2f}s")

    # ========== 数据切分 ==========
    n_jobs = 4
    list_data = np.array_split(X_test, n_jobs)

    print(f"\n数据切分为 {n_jobs} 个部分")
    for i, data in enumerate(list_data):
        print(f"  Part {i+1}: {data.shape[0]} 行")

    # ========== 方案1: ProcessPoolExecutor (原始方案) ==========
    print("\n" + "=" * 60)
    print("方案1: ProcessPoolExecutor (原始方案)")
    print("=" * 60)

    start = time.time()
    pool = ProcessPoolExecutor(max_workers=n_jobs)
    list_shap_interaction_ppe = list(pool.map(
        cal_shap_interaction_values, 
        [explainer] * n_jobs,  # ❌ 问题：重复序列化
        list_data
    ))
    arr3d_result_ppe = np.vstack(list_shap_interaction_ppe)
    time_ppe = time.time() - start

    pool.shutdown(wait=True)

    print(f"耗时: {time_ppe:.2f}s")
    print(f"结果形状: {arr3d_result_ppe.shape}")
    print(f"内存占用: {arr3d_result_ppe.nbytes / 1024 / 1024:.2f} MB")

    # ========== 方案2: ProcessPoolExecutor (改进方案) ==========
    print("\n" + "=" * 60)
    print("方案2: ProcessPoolExecutor (改进方案)")
    print("=" * 60)

    from functools import partial

    start = time.time()
    cal_func = partial(cal_shap_interaction_values, explainer)
    pool = ProcessPoolExecutor(max_workers=n_jobs)
    list_shap_interaction_ppe_opt = list(pool.map(cal_func, list_data))
    arr3d_result_ppe_opt = np.vstack(list_shap_interaction_ppe_opt)
    time_ppe_opt = time.time() - start

    pool.shutdown(wait=True)

    print(f"耗时: {time_ppe_opt:.2f}s")
    print(f"结果形状: {arr3d_result_ppe_opt.shape}")

    # ========== 方案3: Joblib 多进程 ==========
    print("\n" + "=" * 60)
    print("方案3: Joblib 多进程 (loky)")
    print("=" * 60)

    start = time.time()
    list_shap_interaction_joblib_proc = Parallel(
        n_jobs=n_jobs,
        backend='loky',
        verbose=10
    )(
        delayed(cal_shap_interaction_values)(explainer, data_chunk)
        for data_chunk in list_data
    )
    arr3d_result_joblib_proc = np.vstack(list_shap_interaction_joblib_proc)
    time_joblib_proc = time.time() - start

    print(f"耗时: {time_joblib_proc:.2f}s")
    print(f"结果形状: {arr3d_result_joblib_proc.shape}")

    # ========== 方案4: Joblib 多线程 ==========
    print("\n" + "=" * 60)
    print("方案4: Joblib 多线程 (threading)")
    print("=" * 60)

    start = time.time()
    list_shap_interaction_joblib_thread = Parallel(
        n_jobs=n_jobs,
        backend='threading',
        verbose=10,
        max_nbytes=None
    )(
        delayed(cal_shap_interaction_values)(explainer, data_chunk)
        for data_chunk in list_data
    )
    arr3d_result_joblib_thread = np.vstack(list_shap_interaction_joblib_thread)
    time_joblib_thread = time.time() - start

    print(f"耗时: {time_joblib_thread:.2f}s")
    print(f"结果形状: {arr3d_result_joblib_thread.shape}")

    # ========== 单线程基准 ==========
    print("\n" + "=" * 60)
    print("基准: 单线程顺序执行")
    print("=" * 60)

    start = time.time()
    list_shap_interaction_single = [
        cal_shap_interaction_values(explainer, data_chunk)
        for data_chunk in list_data
    ]
    arr3d_result_single = np.vstack(list_shap_interaction_single)
    time_single = time.time() - start

    print(f"耗时: {time_single:.2f}s")
    print(f"结果形状: {arr3d_result_single.shape}")

    # ========== 结果验证 ==========
    print("\n" + "=" * 60)
    print("结果验证（确保所有方案结果相同）")
    print("=" * 60)

    # 检查数值是否接近
    tolerance = 1e-10

    check_1_2 = np.allclose(arr3d_result_ppe, arr3d_result_ppe_opt, atol=tolerance)
    check_1_3 = np.allclose(arr3d_result_ppe, arr3d_result_joblib_proc, atol=tolerance)
    check_1_4 = np.allclose(arr3d_result_ppe, arr3d_result_joblib_thread, atol=tolerance)
    check_1_5 = np.allclose(arr3d_result_ppe, arr3d_result_single, atol=tolerance)

    print(f"PPE原始 vs PPE改进:      {'✓ 通过' if check_1_2 else '✗ 失败'}")
    print(f"PPE原始 vs Joblib进程:   {'✓ 通过' if check_1_3 else '✗ 失败'}")
    print(f"PPE原始 vs Joblib线程:   {'✓ 通过' if check_1_4 else '✗ 失败'}")
    print(f"PPE原始 vs 单线程:       {'✓ 通过' if check_1_5 else '✗ 失败'}")

    # ========== 性能对比总结 ==========
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)

    results = {
        'ProcessPoolExecutor (原始)': time_ppe,
        'ProcessPoolExecutor (改进)': time_ppe_opt,
        'Joblib 多进程 (loky)': time_joblib_proc,
        'Joblib 多线程 (threading)': time_joblib_thread,
        '单线程基准': time_single,
    }

    # 按耗时排序
    sorted_results = sorted(results.items(), key=lambda x: x[1])

    print(f"\n{'方案':<35} {'耗时(s)':<12} {'加速比':<12}")
    print("-" * 65)

    for i, (method, time_val) in enumerate(sorted_results):
        speedup = time_single / time_val
        rank = f"#{i+1}"
        print(f"{rank} {method:<32} {time_val:<12.2f} {speedup:<12.2f}x")

    # ========== 可视化 ==========
    print("\n" + "=" * 60)
    print("绘制性能对比图")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 柱状图
    ax1 = axes[0]
    methods = list(results.keys())
    times = list(results.values())
    colors = ['#FF6B6B', '#FF8C42', '#4ECDC4', '#45B7D1', '#95E1D3']

    bars = ax1.bar(range(len(methods)), times, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('耗时 (秒)', fontsize=12, fontweight='bold')
    ax1.set_title('SHAP 相互作用计算耗时对比\n(随机森林回归)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 加速比图
    ax2 = axes[1]
    speedups = [time_single / t for t in times]
    bars2 = ax2.bar(range(len(methods)), speedups, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('相对于单线程的加速倍数', fontsize=12, fontweight='bold')
    ax2.set_title('加速比对比\n(相对于单线程)', fontsize=13, fontweight='bold')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='基准线 (1x)')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)

    # 添加数值标签
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('shap_interaction_performance_rf.png', dpi=300, bbox_inches='tight')
    print("✓ 性能对比图已保存: shap_interaction_performance_rf.png")
    plt.show()

    # ========== 详细性能分析 ==========
    print("\n" + "=" * 60)
    print("详细性能分析")
    print("=" * 60)

    print(f"\n数据集信息:")
    print(f"  - 总样本数: {n_samples}")
    print(f"  - 特征数: {n_features}")
    print(f"  - 信息特征数: {n_informative}")
    print(f"  - 训练集: {X_train.shape}")
    print(f"  - 测试集: {X_test.shape}")
    print(f"  - 输出形状 (相互作用): {arr3d_result_ppe.shape}")
    print(f"  - 单个相互作用矩阵大小: {arr3d_result_ppe[0].nbytes / 1024:.2f} KB")
    print(f"  - 总输出大小: {arr3d_result_ppe.nbytes / 1024 / 1024:.2f} MB")

    print(f"\n时间分解:")
    print(f"  - 模型训练: {train_time:.2f}s")
    print(f"  - Explainer 创建: {explainer_time:.2f}s")
    print(f"  - SHAP 计算 (最优): {time_joblib_thread:.2f}s")
    print(f"  - 总耗时: {train_time + explainer_time + time_joblib_thread:.2f}s")

    print(f"\n推荐方案排序:")
    print(f"  1️⃣  最快: Joblib 多线程 ({time_joblib_thread:.2f}s, {time_single/time_joblib_thread:.2f}x)")
    print(f"  2️⃣  稳定: Joblib 多进程 ({time_joblib_proc:.2f}s, {time_single/time_joblib_proc:.2f}x)")
    print(f"  3️⃣  改进: PPE改进方案 ({time_ppe_opt:.2f}s, {time_single/time_ppe_opt:.2f}x)")
    print(f"  4️⃣  避免: PPE原始方案 ({time_ppe:.2f}s, {time_single/time_ppe:.2f}x)")

    print(f"\n性能收益:")
    print(f"  - Joblib线程 vs PPE原始: {time_ppe/time_joblib_thread:.2f}x 快")
    print(f"  - Joblib线程 vs 单线程: {time_single/time_joblib_thread:.2f}x 快")
    print(f"  - Joblib线程 vs PPE改进: {time_ppe_opt/time_joblib_thread:.2f}x 快")

    print(f"\n模型性能:")
    print(f"  - R² 分数: {r2_score:.4f}")
    print(f"  - 特征重要性前3: {sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:3]}")

    print("\n" + "=" * 60)
    print("✓ 测试完成！")
    print("=" * 60)
