import numpy as np
import warnings
from utils import remove_bad_rows
from data_Loss import pairwise_independence, nonpredictability_loss_lgb, rf_mse, ridge_mse, lgb_mse, nonpredictability_loss, nonpredictability_loss_lgb_r2
from tqdm import tqdm

# 忽略与power运算相关的警告
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
def evaluate(individual,c_mask, **kwargs):
    """
    评估个体的性能
    """
    # 从 kwargs 中获取所需参数
    W_p_expanded_repeat = kwargs['W_p_expanded_repeat'] # W_p 扩展矩阵
    N = kwargs['N']  # null space 矩阵
    idx_train = kwargs['idx_train'] # 训练集索引
    idx_test = kwargs['idx_test']   # 测试集索引
    X = kwargs['X']  # 自变量
    y = kwargs['y']  # 因变量
    r = kwargs['r']  # N 的列数

    c_vector = individual.reshape(-1, 1)  # (c_num * r, 1)
    c_mask = c_mask.reshape(-1, 1)
    c_num = c_mask.shape[0]
    # 得到最后的x
    repeated_c_vector = np.repeat(c_mask, r, axis=0)
    masked_c_vector = c_vector * repeated_c_vector # c_num * r
    masked_c_vector = masked_c_vector.reshape(r, -1, order='F')
    x = W_p_expanded_repeat + N @ masked_c_vector   # n* c_num

    preds = []
    for i in range(c_num):
        xi = x[:, i].reshape(-1, 1)  # n x 1
        pred_i = np.prod(X ** xi.T, axis=1)  # m x 1
        preds.append(pred_i)
    preds = np.column_stack(preds)  # m x c_num

    preds_train, y_train = remove_bad_rows(preds[idx_train], y[idx_train])
    preds_test, y_test = remove_bad_rows(preds[idx_test], y[idx_test])

    # 计算各个自变量之间的相关性矩阵，取最大值
    # corr_matrix = np.corrcoef(preds_train.T)
    # corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    # # 将对角线元素设置为0，然后按每一列取最大值
    # np.fill_diagonal(corr_matrix, 0)
    # correlation = np.max(np.abs(corr_matrix), axis=0)

    # 目标1: 数据损失
    preds_train = preds_train * c_mask.T
    preds_test = preds_test * c_mask.T
    mse, r2, feature_importances = ridge_mse(preds_train, y_train, preds_test, y_test)
    
    # 目标2: 正交损失
    # k = int(np.sum(c_mask))  # 激活的无量纲数数量

    # non_zero_cols = np.any(preds_train != 0, axis=0)
    # preds_train = preds_train[:, non_zero_cols]
    pi_with_y = np.column_stack([preds_train, y_train])
    # 捕捉这个的相关性
    corr_matrix_pi_y = np.corrcoef(pi_with_y.T)
    corr_matrix_pi_y = np.nan_to_num(corr_matrix_pi_y, nan=0.0)
    np.fill_diagonal(corr_matrix_pi_y, 0)
    corr_matrix_y = corr_matrix_pi_y[:-1, -1].copy()  # 最后一列是 y 的相关性
    corr_matrix_pi = corr_matrix_pi_y[:-1, :-1]
    correlation = np.max(np.abs(corr_matrix_pi), axis=0)
    orth_loss = np.mean(correlation) - np.mean(np.abs(corr_matrix_y))
    # orth_loss = pairwise_independence(pi_with_y)

    # if k < 2:
    #     # return mse, feature_importances, 0, correlation
    #     # non_zero_cols = np.any(preds_train != 0, axis=0)
    #     # preds_test_single = preds_test[:, non_zero_cols]

    #     # # 计算不可预测性：因为只有一列，没有其他特征可以预测
    #     # # 直接用该列的方差作为“不可预测性”
    #     # Zi_test = preds_test_single
    #     # orth_loss = np.var(Zi_test)  # 或者 np.mean((Zi_test - Zi_test.mean())**2)
    #     # 取负号保持原逻辑（越大越不可预测） 我最小化这一列和y的损失不就行吗？ 使用lgb来得到这个mse 然后最小化。
    #     orth_loss = 0
    # else:

    #     non_zero_cols = np.any(preds_train != 0, axis=0)
    #     preds_train = preds_train[:, non_zero_cols]

    #     non_zero_cols = np.any(preds_test != 0, axis=0)
    #     preds_test = preds_test[:, non_zero_cols]

    #     correlation_copy = correlation[correlation != 0]

    #     n_samples = preds_train.shape[0]
    #     sample_idx = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    #     preds_sampled = preds_train[sample_idx, :]

    #     n_samples = preds_test.shape[0]
    #     sample_idx = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    #     preds_test_sampled = preds_test[sample_idx, :]

    #     orth_loss = -nonpredictability_loss_lgb(preds_sampled, preds_test_sampled, correlation_copy) 

    # # non_zero_cols = np.any(preds_train != 0, axis=0)
    # # preds_train = preds_train[:, non_zero_cols]

    # # orth_loss = calculate_unpredictability_loss(preds_train, y_train, c_mask)

    return mse, feature_importances, orth_loss, correlation

def evaluate_population(individuals, c_masks, **kwargs):
    """
    评估整个种群的性能
    """
    results = []
    feature_importances = []
    correlations = []
    for idx in range(individuals.shape[0]):
        individual = individuals[idx]
        c_mask = c_masks[idx]
        mse, feature_importance, orth_loss, correlation = evaluate(individual, c_mask, **kwargs)
        results.append([mse, orth_loss])
        feature_importances.append(feature_importance)
        correlations.append(correlation)
    return np.array(results), np.array(feature_importances), np.array(correlations)