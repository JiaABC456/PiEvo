import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
def sklearn_split_method_with_indices(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    使用 sklearn 的分割方法，并返回对应的原始索引
    
    返回:
        X_train, X_val, X_test: 特征数据
        y_train, y_val, y_test: 目标数据
        train_idx, val_idx, test_idx: 对应的原始索引
    """
    # 生成原始索引
    original_indices = np.arange(len(y))
    
    # 第一次分割：训练+验证集 vs 测试集
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X, y, original_indices, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=None
    )
    
    # 第二次分割：训练集 vs 验证集
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val, y_train_val, idx_train_val,
        test_size=val_size/(1-test_size), 
        random_state=random_state, 
        stratify=None
    )
    
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            idx_train, idx_val, idx_test)


def remove_bad_rows(preds, *other_arrays):
    """
    删除包含 NaN、Inf 和超过 float32 范围的行
    """
    # 检查 NaN 和 Inf
    bad_mask = np.isnan(preds) | np.isinf(preds)
    
    # 检查是否超过 float32 范围
    float32_max = 3.4e38
    bad_mask = bad_mask | (np.abs(preds) > float32_max)
    # 找出坏的行
    bad_rows = np.any(bad_mask, axis=1)
    good_rows = ~bad_rows
    
    # 清理
    clean_preds = preds[good_rows]
    
    # 清理其他数组
    if other_arrays:
        clean_others = [arr[good_rows] for arr in other_arrays]
        return (clean_preds, *clean_others)
    
    return clean_preds

def sign_match_probability(x, sign_vec, eps=1e-8):
    """
    统计单个 x 与 sign_vec 的符号一致率

    Parameters
    ----------
    x : array-like, shape (n_features,)
        反推出的指数向量
    sign_vec : array-like, shape (n_features,)
        相关性得到的符号先验 {-1, 0, 1}
    eps : float
        数值稳定阈值，避免 0 附近抖动

    Returns
    -------
    match_rate : float
        符号一致的比例（只在 sign_vec != 0 的维度上统计）
    """
    x = x.flatten()
    sign_vec = sign_vec.flatten()

    valid_idx = np.where(sign_vec != 0)[0]
    if len(valid_idx) == 0:
        return np.nan  # 没有可比的维度

    matches = 0
    for i in valid_idx:
        if x[i] > eps and sign_vec[i] > 0:
            matches += 1
        elif x[i] < -eps and sign_vec[i] < 0:
            matches += 1

    return matches / len(valid_idx)

def plot_loss_distribution_with_init(results, sizes):
    """
    results: ndarray, shape (pop_size, k)
             第0列：MSE，第2列：正交性损失
    sizes: tuple/list (n1, n2, n3)
           三种初始化方法对应的个体数量
    """
    mse = results[:, 0]
    orth = results[:, 1]

    n1, n2 = sizes
    idx1 = slice(0, n1)
    idx2 = slice(n1, n1 + n2)

    plt.figure(figsize=(7, 5))

    plt.scatter(mse[idx1], orth[idx1],
                label="Sign-aware Initialization",
                marker="o", alpha=0.7)

    plt.scatter(mse[idx2], orth[idx2],
                label="Random Initialization",
                marker="^", alpha=0.7)

    plt.xlabel("Data Loss (MSE)")
    plt.ylabel("Orthogonality Loss")
    plt.title("Initialization Strategy Comparison")
    plt.legend()
    plt.grid(True)
    # plt.xlim(0, 500000)
    plt.tight_layout()
    plt.show()

def fast_non_dominated_sort(P, N):
    pop_size = P.shape[0]
    n = np.zeros(pop_size, dtype=int)   # 被支配数
    S = [[] for _ in range(pop_size)]   # 支配个体列表
    rank = np.full(pop_size, np.inf, dtype=float)  # 初始未分配

    # ------------------ 计算支配关系 ------------------
    for p in range(pop_size):
        for q in range(pop_size):
            if p == q:
                continue
            if np.all(P[p] <= P[q]) and np.any(P[p] < P[q]):  # p 支配 q
                S[p].append(q)
            elif np.all(P[q] <= P[p]) and np.any(P[q] < P[p]):  # q 支配 p
                n[p] += 1

    # ------------------ 第一前沿 ------------------
    current_front = [i for i in range(pop_size) if n[i] == 0]
    for i in current_front:
        rank[i] = 1

    fronts = [current_front]
    assigned = len(current_front)
    f_no = 1

    # ------------------ 迭代生成前沿 ------------------
    while assigned < N and len(fronts[f_no-1]) > 0:
        next_front = []
        for p in fronts[f_no-1]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = f_no + 1
                    next_front.append(q)
        f_no += 1
        fronts.append(next_front)
        assigned += len(next_front)

    MaxFNo = f_no
    return rank, MaxFNo

def crowding_distance(PopObj, FrontNo=None):
    """
    Calculate crowding distances of solutions for each non-dominated front.

    Parameters
    ----------
    PopObj : ndarray, shape (N, M)
        Objective values of the population (rows=solutions, cols=objectives)
    FrontNo : ndarray, shape (N,), optional
        Front number of each solution. If None, all solutions are treated as one front.

    Returns
    -------
    CrowdDis : ndarray, shape (N,)
        Crowding distance of each solution (inf for boundary solutions)
    """

    N, M = PopObj.shape
    if FrontNo is None:
        FrontNo = np.ones(N, dtype=int)

    CrowdDis = np.zeros(N)
    Fronts = np.setdiff1d(np.unique(FrontNo), np.inf)  # all finite fronts

    for f in Fronts:
        idx = np.where(FrontNo == f)[0]  # indices of solutions in front f
        Fmax = np.max(PopObj[idx, :], axis=0)
        Fmin = np.min(PopObj[idx, :], axis=0)

        for m in range(M):
            # sort indices of this front based on objective m
            sorted_idx = idx[np.argsort(PopObj[idx, m])]
            # boundary solutions get infinite distance
            CrowdDis[sorted_idx[0]] = np.inf
            CrowdDis[sorted_idx[-1]] = np.inf

            # compute crowding distance for internal solutions
            for j in range(1, len(sorted_idx) - 1):
                if Fmax[m] - Fmin[m] > 1e-12:  # avoid division by zero
                    distance = (PopObj[sorted_idx[j + 1], m] - PopObj[sorted_idx[j - 1], m]) / (Fmax[m] - Fmin[m])
                    CrowdDis[sorted_idx[j]] += distance

    return CrowdDis

def plot_fronts(results, rank):
    """
    可视化非支配排序的前沿分布（适用于二维目标）

    Parameters
    ----------
    results : ndarray, shape (N, 2)
        目标值矩阵，每行对应一个个体，每列对应一个目标
    rank : ndarray, shape (N,)
        每个个体所属前沿编号（-1 表示未被排序）
    """
    # 只取已排前沿的个体
    fronts = np.unique(rank[rank >= 0])
    colors = plt.cm.get_cmap('tab10', len(fronts))  # 为每个前沿分配颜色

    plt.figure(figsize=(7,5))

    for i, f in enumerate(fronts):
        idx = np.where(rank == f)[0]
        plt.scatter(results[idx,0], results[idx,1], 
                    label=f'Front {f}', s=60, alpha=0.7,
                    color=colors(i))

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Non-dominated Sorting (Fronts)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_population(results, c_masks, save_path='population_plot.png'):
    """
    可视化当前种群的目标分布，用颜色标识激活无量纲数个数（适用于二维目标）

    Parameters
    ----------
    results : ndarray, shape (N, 2)
        目标值矩阵，每行对应一个个体，每列对应一个目标
    c_masks : ndarray, shape (N, c_num)
        每个个体的掩码矩阵，1 表示激活，0 表示未激活
    save_path : str
        保存图像的路径和文件名，默认为 'population_plot.png'
    """
    # 确保保存目录存在
    save_dir = Path(save_path).parent
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算每个解激活的无量纲数个数
    num_active = np.sum(c_masks, axis=1)
    
    plt.figure(figsize=(8, 5))
    
    # 使用激活数作为颜色映射
    scatter = plt.scatter(results[:, 0], results[:, 1], 
                          c=num_active, cmap='YlOrRd', 
                          s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Active Π', fontsize=10)
    
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Population Objective Distribution (Color: Active Π Count)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_front(results, true_pareto_front):
    """
    可视化当前种群的 Pareto 前沿与真实前沿对比（适用于二维目标）

    Parameters
    ----------
    results : ndarray, shape (N, 2)
        当前种群的目标值矩阵
    true_pareto_front : ndarray, shape (M, 2)
        真实的 Pareto 前沿目标值矩阵
    """
    plt.figure(figsize=(7,5))

    # 绘制当前种群
    plt.scatter(results[:,0], results[:,1], 
                label='Current Population', s=40, alpha=0.6, color='blue')

    # 绘制真实前沿
    plt.plot(true_pareto_front[:,0], true_pareto_front[:,1], 
             label='True Pareto Front', color='red', linewidth=2)

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Front Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def TS(fitness):
    """
    Binary Tournament Selection based on fitness values.

    Parameters
    ----------
    fitness : ndarray, shape (n,)
        Fitness values of the individuals.

    Returns
    -------
    selected_indices : list
        Indices of the selected individuals.
    """
    if len(fitness) == 0:
        return []
    else:
        return tournament_selection(2,1,fitness)

def GAreal(Parent1, Parent2, importance, lower, upper, c_num, r, proC = 1, disC = 20, proM = 1, disM = 20):
    """
    Genetic operators for real-valued variables: simulated binary crossover and polynomial mutation.
    
    Parameters
    ----------
    Parent1 : ndarray, shape (N, D)
        First parent population (N individuals, D variables each).
    Parent2 : ndarray, shape (N, D)
        Second parent population (same shape as Parent1).
    lower : ndarray, shape (D,)
        Lower bounds of variables.
    upper : ndarray, shape (D,)
        Upper bounds of variables.
    proC : float
        Crossover probability (0~1).
    disC : float
        Distribution index for crossover (usually 5~20).
    proM : float
        Mutation probability (0~1).
    disM : float
        Distribution index for mutation (usually 5~50).
    
    Returns
    -------
    Offspring : ndarray, shape (N, D)
        Offspring population after crossover and mutation.
    """
    N, D = Parent1.shape
    
    # Simulated binary crossover (SBX)
    beta = np.zeros((N, D))
    mu = np.random.rand(N, D)
    
    mask = mu <= 0.5
    beta[mask] = (2 * mu[mask]) ** (1 / (disC + 1))
    beta[~mask] = (2 - 2 * mu[~mask]) ** (-1 / (disC + 1))
    
    # Randomly flip signs
    beta *= (-1) ** np.random.randint(0, 2, size=(N, D))
    beta[np.random.rand(N, D) < 0.5] = 1
    
    # Apply crossover probability
    mask_proC = np.random.rand(N, 1) > proC
    beta[mask_proC.repeat(D, axis=1)] = 1
    
    Offspring = (Parent1 + Parent2) / 2 + beta * (Parent1 - Parent2) / 2
    
    # Polynomial mutation
    Lower = np.tile(lower, (N, 1))
    Upper = np.tile(upper, (N, 1))
    
    Site = np.random.rand(N, D) < proM / D
    mu = np.random.rand(N, D)
    
    # Mutation for mu <= 0.5
    temp = Site & (mu <= 0.5)
    Offspring[temp] += (Upper[temp] - Lower[temp]) * \
        ((2 * mu[temp] + (1 - 2 * mu[temp]) * \
          (1 - (Offspring[temp] - Lower[temp]) / (Upper[temp] - Lower[temp])) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
    
    # Mutation for mu > 0.5
    temp = Site & (mu > 0.5)
    Offspring[temp] += (Upper[temp] - Lower[temp]) * \
        (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * \
             (1 - (Upper[temp] - Offspring[temp]) / (Upper[temp] - Lower[temp])) ** (disM + 1)) ** (1 / (disM + 1)))

    # for i in range(N):  # 遍历每个个体
    #     # Reshape 成 (c_num, r)
    #     c_individual = Offspring[i].reshape(c_num, r)
    #     # 正交性变异
    #     c_individual = diversify_c_importance_aware(c_individual, importance[i])
    #     # 写回 Offspring
    #     Offspring[i] = c_individual.reshape(-1)
    Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
    
    return Offspring

def diversify_c_importance_aware(
    c,
    importance,
    base_alpha=0.2,
    p_mut=0.5,
    eps=1e-8
):
    """
    根据特征重要性自适应正交性变异
    """
    c_new = c.copy()
    c_num, r = c.shape

    # 归一化重要性（越大越重要）
    imp = importance / (np.max(importance) + eps)

    # 变异概率：重要性越低，越容易变异
    p = p_mut * (1 - imp)

    # 选中要变异的 c
    mask = np.random.rand(c_num) < p
    if np.sum(mask) < 2:
        return c_new

    c_sel = c_new[mask]

    # 正交方向
    c_norm = np.linalg.norm(c_sel, axis=1, keepdims=True)
    c_unit = c_sel / (c_norm + eps)

    Q, _ = np.linalg.qr(c_unit.T)
    Q = Q.T[:c_sel.shape[0]]

    # 每个 c 有不同的旋转强度
    alpha = base_alpha * (1 - imp[mask])[:, None]

    c_new[mask] = c_sel + alpha * Q
    return c_new

def tournament_selection(K, N, *fitness_list):
    """
    K-tournament selection : 选择 最小对应的索引

    Parameters
    ----------
    K : int
        Tournament size
    N : int
        Number of parents to select
    fitness_list : list of np.ndarray
        Fitness arrays, shape (pop_size,)

    Returns
    -------
    index : np.ndarray, shape (N,)
        Selected indices
    """
    # 确保 fitness 是列向量
    fitness_list = [f.reshape(-1, 1) for f in fitness_list]

    # 拼接多目标 fitness
    Fit = np.hstack(fitness_list)

    # 计算唯一行，并得到原位置索引
    _, Loc = np.unique(Fit, axis=0, return_inverse=True)

    # 按行排序 Fit
    rank = np.lexsort(Fit[:, ::-1].T)  # MATLAB sortrows 逆序列 -> np.lexsort
    rank_array = np.empty_like(rank)
    rank_array[rank] = np.arange(len(rank))  # MATLAB rank(rank)=sort index

    # 随机选择 K 个候选人，生成 shape (K, N)
    Parents = np.random.randint(0, Fit.shape[0], size=(K, N))

    # 取候选人的 rank
    candidate_rank = rank_array[Loc[Parents]]  # shape (K, N)

    # 选择每列最优（最小 rank）
    best_in_column = np.argmin(candidate_rank, axis=0)

    # 转换为原索引
    index = Parents[best_in_column, np.arange(N)]

    return index

def visualize_full_corr_matrix(corr_matrix, c_mask=None, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1: 相关系数矩阵热图
    ax1 = axes[0]
    im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title('Correlation Matrix Heatmap')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Feature Index')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 在热图上直接添加数值
    n = corr_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f'{corr_matrix[i,j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if abs(corr_matrix[i,j]) > 0.5 else 'black',
                    fontsize=8)
    
    # 如果有c_mask，标记激活特征
    if c_mask is not None:
        c_mask = c_mask.flatten()
        active_indices = np.where(c_mask)[0]
        for idx in active_indices:
            ax1.add_patch(plt.Rectangle((idx-0.5, idx-0.5), 1, 1, 
                                      fill=False, edgecolor='lime', linewidth=1.5))
    
    # 子图2: 相关系数绝对值分布直方图
    ax2 = axes[1]
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    corr_values = corr_matrix[mask]
    abs_corr = np.abs(corr_values)
    
    ax2.hist(abs_corr, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=np.mean(abs_corr), color='red', linestyle='--', 
                label=f'Mean: {np.mean(abs_corr):.3f}')
    ax2.axvline(x=np.median(abs_corr), color='green', linestyle='--', 
                label=f'Median: {np.median(abs_corr):.3f}')
    
    ax2.set_title('Distribution of Correlation Coefficients')
    ax2.set_xlabel('|Correlation|')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存或显示图片
    if save_path is not None:
        # 确保保存目录存在
        save_dir = Path(save_path).parent
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"相关系数矩阵图已保存到: {save_path}")
    else:
        plt.show()
    
    return fig



def create_labels(omega, variables):
    labels = []
    for row in omega:
        positive_part = ''
        negative_part = ''
        for i, value in enumerate(row):
            value = float(value)  # Safe scalar cast
            value = np.round(value, 2)  # Round to two decimal places
            if value > 0:
                if positive_part == '':
                    positive_part = f"{variables[i]}^{{{value}}}"
                else:
                    positive_part += f" \\cdot {variables[i]}^{{{value}}}"
            elif value < 0:
                if negative_part == '':
                    negative_part = f"{variables[i]}^{{{-value}}}"
                else:
                    negative_part += f" \\cdot {variables[i]}^{{{-value}}}"
        if negative_part == '':
            labels.append(f"${positive_part}$")
        elif positive_part == '':
            labels.append(f"$\\frac{{1}}{{{negative_part}}}$")
        else:
            labels.append(f"$\\frac{{{positive_part}}}{{{negative_part}}}$")
    return labels


def evaluate_individual(individual, c_mask, W_p, N, X_train, y_train, X_test, y_test, model):
    
    # 从 kwargs 中获取所需参数
    W_p_expanded_repeat = np.repeat(W_p, c_mask.shape[0], axis=1)
    r = N.shape[1]  # N 的列数

    c_vector = individual.reshape(-1, 1)  # (c_num * r, 1)
    c_mask = c_mask.reshape(-1, 1)
    c_num = c_mask.shape[0]
    # 得到最后的x
    repeated_c_vector = np.repeat(c_mask, r, axis=0)
    masked_c_vector = c_vector * repeated_c_vector # c_num * r
    masked_c_vector = masked_c_vector.reshape(r, -1, order='F')
    x = W_p_expanded_repeat + N @ masked_c_vector   # n* c_num
    
    # 训练集
    preds = []
    for i in range(c_num):
        xi = x[:, i].reshape(-1, 1)  # n x 1
        pred_i = np.prod(X_train ** xi.T, axis=1)  # m x 1
        preds.append(pred_i)
    preds = np.column_stack(preds)  # m x c_num
    preds, y_train = remove_bad_rows(preds, y_train)

    # 训练
    preds = preds * c_mask.T
    model.fit(preds, y_train)

    # 测试集
    preds = []
    for i in range(c_num):
        xi = x[:, i].reshape(-1, 1)  # n x 1
        pred_i = np.prod(X_test ** xi.T, axis=1)  # m x 1
        preds.append(pred_i)
    preds = np.column_stack(preds)  # m x c_num
    preds, y_test = remove_bad_rows(preds, y_test)

    # 测试集预测
    preds = preds * c_mask.T
    y_pred = model.predict(preds)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, model.feature_importances_

def evaluate_individual_TI_PI(input_PI_train, y_train, input_PI_test, y_test, model):
    """
    使用直接的指数矩阵 individual（形状: n_features x n_pi）构造 Π 特征并评估模型。

    individual: ndarray, shape (n_features, n_pi)
    c_mask: array-like, shape (n_pi,), 1 表示该 Π 激活
    W_p, N: 保留占位参数（未使用）
    """

    model.fit(input_PI_train,y_train)

    y_pred = model.predict(input_PI_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, getattr(model, 'feature_importances_', None)



# =============== 噪声注入辅助函数 ===============
def apply_noise(X, y, x_noise_frac=0.0, y_noise_frac=0.0, seed=0):
    """对输入/输出加入高斯噪声。

    x_noise_frac: 按列标准差的比例添加噪声
    y_noise_frac: 按 y 标准差的比例添加噪声
    seed:         随机种子，便于重复实验
    """
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()
    y_noisy = y.copy()

    if x_noise_frac > 0:
        col_std = np.std(X_noisy, axis=0, keepdims=True)
        col_std[col_std == 0] = 1e-12
        X_noisy = X_noisy + rng.normal(0.0, x_noise_frac * col_std, size=X_noisy.shape)

    if y_noise_frac > 0:
        y_std = np.std(y_noisy)
        if y_std == 0:
            y_std = 1e-12
        y_noisy = y_noisy + rng.normal(0.0, y_noise_frac * y_std, size=y_noisy.shape)

    return X_noisy, y_noisy