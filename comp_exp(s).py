from numpy.linalg import matrix_rank, inv
from sklearn.metrics import mean_squared_error, r2_score
from generate_data import *
from utils import evaluate_individual_TI_PI, remove_bad_rows, sklearn_split_method_with_indices, apply_noise, evaluate_individual
from sklearn.ensemble import RandomForestRegressor
from scipy.linalg import null_space
import sys
import os
import json
import numpy as np
import pandas as pd
import time


def DEA_run(config, X_train, y_train, X_test, y_test, variable_names):
    from DEA import DEA
    dea = DEA(config, data=(X_train, y_train, variable_names))
    t0 = time.time()
    dea.run()
    analysis_runtime_sec = time.time() - t0
    # 得到性能最好或者最稀疏的个体
    sparsity = np.sum(dea.c_masks, axis=1)          # shape: (pop_size,)
    mses = dea.results[:, 0]
    min_sparse = sparsity.min()
    candidate_idx = np.where(sparsity == min_sparse)[0]
    best_idx = candidate_idx[np.argmin(mses[candidate_idx])]

    # 模型建立
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    mse, r2, feature_importances = evaluate_individual(
        dea.individuals[best_idx],
        dea.c_masks[best_idx],
        dea.W_p,
        dea.N,
        X_train, 
        y_train,
        X_test,
        y_test,
        rf
    )

    # 模型建立
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    baseline_mse = mean_squared_error(y_test, y_pred)
    baseline_r2 = r2_score(y_test, y_pred)

    return {
        'algorithm': 'DEA',
        'dataset': dea.config.get("DATA_PATH", "unknown"),
        'number of dimensions': np.sum(dea.c_masks[best_idx]),
        'Best_Individual_MSE': mse,
        'Best_Individual_R2': r2,
        'Feature_Importances': feature_importances,
        'Baseline_MSE': baseline_mse,
        'Baseline_R2': baseline_r2,
        'analysis_runtime_sec': analysis_runtime_sec
    }

    # DEA 问题 包含几个无量纲数 测试集的mse和R2 随机森林算法的mse和R2 .
    # 如何解决多个解的问题？     

def TI_PI_run(config_file, X_train, y_train, X_test, y_test, variable_names):

    baseline_ti_pi_path = os.path.join(os.path.dirname(__file__), "Baseline", "IT_PI")
    sys.path.append(baseline_ti_pi_path)
    import IT_PI
    # 导入config配置文件
    with open(config_file, 'r') as f:
        config = json.load(f)

    c_num = config.get("C_NUM", 10)
    dim_matrix = np.array(config.get("DIM_MATRIX"))
    num_input = config.get("NUM_INPUT", 1)
    lower_bound = config.get("LOWER_BOUND", -2)
    upper_bound = config.get("UPPER_BOUND", 2)
    estimator = config.get("estimator", "binning")
    estimator_params = config.get("estimator_params", {"num_bins": 20})

    D = dim_matrix[:, :-1]      # k x n
    d = dim_matrix[:, -1].reshape(-1, 1)      # k
    W_p, _, _, _ = np.linalg.lstsq(D, d, rcond=None)  # n x 1
    N = null_space(D)   # shape: (n, r)

    t0 = time.time()
    results = IT_PI.main(    
                    X_train,
                    y_train.reshape(-1, 1),
                    np.asmatrix(N.T),
                    num_input=num_input,
                    estimator=estimator,
                    estimator_params=estimator_params,
                    seed=42,
                    bounds = (lower_bound, upper_bound)
            )
    analysis_runtime_sec = time.time() - t0
    # IT_PI 返回的 input_coef 是一个列表，每个元素是长度为 n_features 的系数向量
    # 将其按列堆叠为 [n_features, num_input]

    # input_PI_train = results["input_PI"]  # shape: (n_features, num_input)
    # input_coef = results["input_coef"]
    # pi_train = results["output_PI"]
    # pi_train = pi_train.reshape(-1)
    # input_coef = np.array(results["input_coef"])

    # input_list = [IT_PI.calc_pi_omega(np.array(omega), X_test) for omega in input_coef]
    # input_PI_test = np.column_stack(input_list)
    # # 移除测试集中包含 NaN 或 Inf 的行
    # finite_mask = np.isfinite(input_PI_test).all(axis=1)
    # input_PI_test = input_PI_test[finite_mask]
    # pi_test = y_test[finite_mask]
    # pi_test = pi_test.reshape(-1)


    input_coef = results["input_coef"]
    pi_train = y_train.reshape(-1)

    input_list_train = []
    for omega in input_coef:
        xi = np.array(omega).reshape(1, -1)
        pred_i = np.prod(X_train*xi, axis=1)
        input_list_train.append(pred_i)
    input_PI_train = np.column_stack(input_list_train)   # shape: (n_samples, num_input)
    input_PI_train, pi_train = remove_bad_rows(input_PI_train, pi_train)

    input_list_test = []
    for omega in input_coef:
        xi = np.array(omega).reshape(1, -1)
        pred_i = np.prod(X_test*xi, axis=1)
        input_list_test.append(pred_i)
    input_PI_test = np.column_stack(input_list_test)   # shape: (n_samples, num_input)
    input_PI_test, pi_test = remove_bad_rows(input_PI_test, y_test.reshape(-1))


    # 评估
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    mse, r2, feature_importances = evaluate_individual_TI_PI(
        input_PI_train,
        pi_train,
        input_PI_test,
        pi_test,
        rf,
    )

    # 模型建立
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    baseline_mse = mean_squared_error(y_test, y_pred)
    baseline_r2 = r2_score(y_test, y_pred)

    return {
        'algorithm': 'TI_PI',
        'dataset': config.get("DATA_PATH", "unknown"),
        'number of dimensions': num_input,
        'Best_Individual_MSE': mse,
        'Best_Individual_R2': r2,
        'Feature_Importances': feature_importances,
        'Baseline_MSE': baseline_mse,
        'Baseline_R2': baseline_r2,
        'analysis_runtime_sec': analysis_runtime_sec
    }


def PyDimension_run(config_file, X_train, y_train, X_test, y_test, variable_names):
    
    # 说你能发现一个,你就真发现一个吗，这个不还是要pca进行降维吗？
    baseline_PyDimension_path = os.path.join(os.path.dirname(__file__), "Baseline", "PyDimension-main")
    sys.path.append(baseline_PyDimension_path)

    from pydimension.dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig
    from pydimension.optimization_discovery import OptimizationDiscoverer, OptimizationDiscoveryConfig
    
    def normalize_data(X,y):
        data = np.concatenate([X, y.reshape(-1,1)], axis=1)
        normalized_data = data.copy()
        for col in range(data.shape[1]):
            max_val = data[:, col].max()
            if max_val != 0:
                normalized_data[:, col] = data[:, col] / max_val
            else:
                normalized_data[:, col] = 0
        # 返回归一化后的X和y
        return normalized_data
    normalized_data = normalize_data(X_train, y_train)

    with open(config_file, 'r') as f:
        config = json.load(f)

    full_matrix = np.array(config.get("DIM_MATRIX"))

    variable_names_copy = variable_names.copy()
    variable_names_copy.extend(["target"])

    configs = DimensionalAnalysisConfig()
    analyzer = DimensionalAnalyzer(configs)
    afterdata, basis_vectors = analyzer.process(normalized_data, full_matrix, variable_names_copy)
    data = analyzer.compute_normalized_lg_pis()

    configs = OptimizationDiscoveryConfig()

    configs.num_linear = config.get("NUM_INPUT", 1)

    t0 = time.time()
    optimizer = OptimizationDiscoverer(configs)
    optimizer.process(data, basis_vectors)
    analysis_runtime_sec = time.time() - t0
    # 关键：手动设置原始参数名称
    optimizer.original_parameter_names = variable_names
    optimizer.basis_vectors = basis_vectors
    
    result = optimizer._construct_discovered_equation()
    
    # 详细输出解释
    # print("\n" + "="*80)
    # print("PyDimension 发现的无量纲关系")
    # print("="*80)
    # for detail in result.get('detailed_equations', []):
    #     print(f"  π_discovered_{detail['index']} gamma向量: {detail['gamma_vector']}")
    #     print(f"  原始参数表达式: {detail.get('expression_original_params', 'N/A')}")
    # print("="*80 + "\n")
    
    # ==================== 测试集评估 ====================
    # 1. 提取所有gamma向量（基向量空间的指数）
    detailed_eqs = result.get('detailed_equations', [])
    gamma_basis = np.column_stack([detail['gamma_vector'] for detail in detailed_eqs])  # (num_basis, num_discovered)
    
    # 2. 将基向量空间的指数转换为原始参数空间的指数
    # basis_vectors 的形状: (num_original_params, num_basis)
    # gamma_original = basis_vectors @ gamma_basis
    exponents_original = basis_vectors @ gamma_basis  # shape: (num_original_params, num_discovered)
    
    # 3. 转换训练集和测试集为无量纲特征（使用原始参数指数）
    # Pi_i = exp(exponents_i^T * log(X)) = 产品(X_j^exponents[j,i])
    log_X_train = np.log(X_train + 1e-10)  # 避免log(0)
    log_X_test = np.log(X_test + 1e-10)
    
    Pi_train = np.exp(log_X_train @ exponents_original)  # shape: (n_train, num_discovered)
    Pi_test = np.exp(log_X_test @ exponents_original)    # shape: (n_test, num_discovered)
    
    # 4. 使用随机森林在无量纲特征上训练和预测
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(Pi_train, y_train)
    y_pred = rf.predict(Pi_test)
    
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    feature_importances = rf.feature_importances_.tolist()
    
    # 5. Baseline（原始特征）
    rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)
    y_pred_baseline = rf_baseline.predict(X_test)
    baseline_mse = mean_squared_error(y_test, y_pred_baseline)
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    
    return {
        'algorithm': 'PyDimension',
        'dataset': config.get("DATA_PATH", "unknown"),
        'Best_Individual_MSE': test_mse,
        'Best_Individual_R2': test_r2,
        'Feature_Importances': feature_importances,
        'Baseline_MSE': baseline_mse,
        'Baseline_R2': baseline_r2,
        'analysis_runtime_sec': analysis_runtime_sec
    }

def Kridge_run(config, X_train, y_train, X_test, y_test, variable_names):
    
    baseline_bucki_path = os.path.join(os.path.dirname(__file__), "Baseline", "bucki-data-main")
    sys.path.append(baseline_bucki_path)

    from src.learning import KRidgeReg
    from src.helper_functions import prettify_results
    from numpy.linalg import matrix_rank

    with open(config_file, 'r') as f:
        config = json.load(f)

    dim_matrix = np.array(config.get("DIM_MATRIX"))
    dim_matrix = dim_matrix[:, :-1]  # 去掉最后一列目标变量

    l1_reg = 1e-3 # Solution is sensitive to L1
    alpha = 1e-4
    kernel = 'rbf'
    gamma = 30
    tol = 0.1
    max_denominator = 10
    num_nondim = X_train.shape[1] - matrix_rank(dim_matrix)
    # Ridge Regression
    K = KRidgeReg(X_train, y_train, dim_matrix, num_nondim=num_nondim, #normalize=normalize,
            l1_reg=l1_reg, alpha=alpha, kernel=kernel, gamma=gamma)
    K.use_test_set = False
    x = K.single_run()

    pi1 = x[:, 0]/x[1, 0]
    prettify_results(pi1, variable_names, tol=tol, max_degree=max_denominator)

    print(x)


def run_experiment(alogorithm, config, X_train, y_train, X_test, y_test, variable_names):
    
    """
    Run the experiment with the given algorithm and configuration.
    """
    instances = alogorithm(config, X_train, y_train, X_test, y_test, variable_names)
    return instances

if __name__ == "__main__":

    methods = [DEA_run, PyDimension_run, TI_PI_run]
    function_datas = [handle_CHF_data, handle_rayleigh_data, handle_Wall_model, handle_Keyhole_data, handle_MHD_data, handle_velocity_trans_data]
    config_files = [
                    "./configs/CHF/config.json",
                    "./configs/Rayleigh/config.json",
                    "./configs/Wall_model/config.json",
                    "./configs/Keyhole/config.json",
                    "./configs/MHD/config.json",
                    "./configs/Velocity_transformation/config.json"] 
    results = []
    for i in range(methods.__len__()):
        for j in range(function_datas.__len__()):
            method = methods[i]
            config_file = config_files[j]
            handle_data_function = function_datas[j]

            print(f"Running {method.__name__} on dataset from {config_file}")
            X, y, variable_names = handle_data_function(config_file)
            # 划分数据集
            X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = sklearn_split_method_with_indices(X, y, test_size=0.1, val_size=0.1)
            X_test = np.vstack([X_val, X_test])
            y_test = np.concatenate([y_val, y_test])

            instance = run_experiment(method, config_file, X_train, y_train, X_test, y_test, variable_names)
            # 保存每次实验结果
            instance["config_file"] = config_file
            instance["method_name"] = method.__name__
            results.append(instance)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("experiment_results.csv", index=False)
        print("Saved experiment results to experiment_results.csv")



    # 噪声场景：无噪声 + 不同级别输入/输出高斯噪声
    noise_scenarios = [
        {"name": "clean", "x_noise_frac": 0.0, "y_noise_frac": 0.0},
        {"name": "x_1pct", "x_noise_frac": 0.01, "y_noise_frac": 0.0},
        {"name": "x_5pct", "x_noise_frac": 0.05, "y_noise_frac": 0.0},
        {"name": "x_10pct", "x_noise_frac": 0.10, "y_noise_frac": 0.0},
        {"name": "y_1pct", "x_noise_frac": 0.0, "y_noise_frac": 0.01},
        {"name": "y_5pct", "x_noise_frac": 0.0, "y_noise_frac": 0.05},
        {"name": "y_10pct", "x_noise_frac": 0.0, "y_noise_frac": 0.10},
        {"name": "xy_5pct", "x_noise_frac": 0.05, "y_noise_frac": 0.05},
    ]

    results = []
    # 加载数据集并在噪声场景上循环
    for noise_idx, noise_cfg in enumerate(noise_scenarios):
        print(f"===== Noise scenario: {noise_cfg['name']} (x_noise={noise_cfg['x_noise_frac']}, y_noise={noise_cfg['y_noise_frac']}) =====")
        for i in range(methods.__len__()):
            for j in range(function_datas.__len__()):
                method = methods[i]
                config_file = config_files[j]
                handle_data_function = function_datas[j]

                print(f"Running {method.__name__} on dataset from {config_file}")
                X, y, variable_names = handle_data_function(config_file)

                # 噪声注入（在切分前保证全量同分布）
                X_noisy, y_noisy = apply_noise(X, y, x_noise_frac=noise_cfg["x_noise_frac"], y_noise_frac=noise_cfg["y_noise_frac"], seed=42 + noise_idx)

                # 划分数据集
                X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = sklearn_split_method_with_indices(X_noisy, y_noisy, test_size=0.1, val_size=0.1)
                X_test = np.vstack([X_val, X_test])
                y_test = np.concatenate([y_val, y_test])

                instance = run_experiment(method, config_file, X_train, y_train, X_test, y_test, variable_names)
                # 保存每次实验结果
                instance["config_file"] = config_file
                instance["method_name"] = method.__name__
                instance["noise_name"] = noise_cfg["name"]
                instance["x_noise_frac"] = noise_cfg["x_noise_frac"]
                instance["y_noise_frac"] = noise_cfg["y_noise_frac"]
                instance["noise_seed"] = 42 + noise_idx
                results.append(instance)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("experiment_results_noise.csv", index=False)
        print("Saved experiment results to experiment_results_noise.csv")

    run_experiment(TI_PI_run, config_file, X_train, y_train, X_test, y_test, variable_names)
    run_experiment(PyDimension_run, config_file, X_train, y_train, X_test, y_tes, variable_names)
    run_experiment(Kridge_run, config_file, X_train, y_train, X_test, y_test, variable_names)