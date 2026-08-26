import warnings
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.special import psi
import scipy.spatial as scispa

warnings.filterwarnings('ignore', category=UserWarning, message='.*not have valid feature names.*')
warnings.filterwarnings('ignore', message='.*result may not be accurate.*')

def rf_mse(
    data,
    y,
    data_test,
    y_test,
    n_estimators=50,
    max_depth=None,
    random_state=42
):
    """
    使用随机森林回归，返回 MSE

    Parameters
    ----------
    data : array-like, shape (m, n)
        自变量矩阵 X
    y : array-like, shape (m,)
        因变量
    n_estimators : int
        随机森林树的数量
    max_depth : int or None
        树最大深度
    random_state : int
        随机种子

    Returns
    -------
    mse : float
        均方误差
    model : RandomForestRegressor
        训练好的随机森林模型（可用于特征重要性或预测）
    """
    
    # 1. 构建随机森林回归器
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    
    # 2. 拟合
    rf.fit(data, y)
    
    # 3. 预测并计算 MSE
    y_pred = rf.predict(data_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # feature_importances = rf.feature_importances_
    return mse, r2, rf.feature_importances_

def ridge_mse(
    data,
    y,
    data_test,
    y_test,
    alpha=1.0,
    fit_intercept=True,
):
    """
    使用 Ridge 回归，返回 MSE 和 R2

    Parameters
    ----------
    data : array-like, shape (m, n)
        训练自变量矩阵
    y : array-like, shape (m,)
        训练因变量
    data_test : array-like, shape (m_test, n)
        测试自变量矩阵
    y_test : array-like, shape (m_test,)
        测试因变量
    alpha : float
        Ridge 正则化系数（越大惩罚越强）
    fit_intercept : bool
        是否拟合截距

    Returns
    -------
    mse : float
        测试集均方误差
    r2 : float
        测试集 R^2
    model : Ridge
        训练好的模型（可选）
    """
    # -------- Ridge 回归 --------
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(data, y)

    # -------- 预测 --------
    y_pred = model.predict(data_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    feature_importance = np.abs(model.coef_)
    return mse, r2, feature_importance


def lgb_mse(data, y, data_test, y_test, n_estimators=50, random_state=0):
    """
    LightGBM回归，速度快，内存占用小
    """
    lgb = LGBMRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=-1,  # 不输出日志
        n_jobs=-1    # 使用所有CPU核心
    )
    
    lgb.fit(data, y)
    y_pred = lgb.predict(data_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, lgb.feature_importances_

def nonpredictability_loss(pred_train):
    """
    计算每列特征被其他特征非线性预测的能力，返回总损失。
    
    参数:
        pred_train: np.ndarray, shape (n_samples, n_features)
    
    返回:
        loss: float, 越大表示特征越难被预测
    """
    n_samples, n_features = pred_train.shape
    total_loss = 0.0
    
    for i in range(n_features):
        Zi = pred_train[:, i]
        Z_others = np.delete(pred_train, i, axis=1)
        
        # 使用非线性回归模型预测 Zi
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        model.fit(Z_others, Zi)
        pred = model.predict(Z_others)
        
        # 预测误差越大，说明 Zi 越不被其他特征预测
        mse = np.mean((Zi - pred) ** 2)
        total_loss += mse
    
    # 平均每个特征的不可预测性
    loss = total_loss / n_features
    return loss


def nonpredictability_loss_lgb(preds_train, preds_test, correlation, n_estimators=50, random_state=0):
    """
    CPU-friendly LGBM 计算每列特征不可预测性 loss
    
    参数:
        preds_train: np.ndarray, shape [n_samples, n_features]
        sample_ratio: 抽取样本比例，用于加速
        subset_cols: 每列只用部分其他特征预测
        n_estimators: LGBM 树的数量
        max_depth: LGBM 树深度
    返回:
        loss: float, 越大表示特征越不可预测
    """
    n_samples, n_features = preds_train.shape
    total_loss = 0.0
    scaler_train = StandardScaler()
    preds_train = scaler_train.fit_transform(preds_train)
    preds_test = scaler_train.transform(preds_test)

    for i in range(n_features):
        Zi = preds_train[:, i]
        Z_others = np.delete(preds_train, i, axis=1)
        
        Zi_test = preds_test[:, i]
        Z_others_test = np.delete(preds_test, i, axis=1)


        # 使用非线性回归模型预测 Zi
        model = LGBMRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1,  # 不输出日志
            n_jobs=-1    # 使用所有CPU核心
        )
        model.fit(Z_others, Zi)
        pred = model.predict(Z_others_test)
        
        # 预测误差越大，说明 Zi 越不被其他特征预测
        mse = np.mean((Zi_test - pred) ** 2)
        total_loss += mse
    
    # 平均每个特征的不可预测性
    loss = total_loss / n_features
    return loss



def nonpredictability_loss_r2(pred_train):
    """
    计算每列特征被其他特征非线性预测的能力，返回总损失（基于 R²）
    
    参数:
        pred_train: np.ndarray, shape (n_samples, n_features)
    
    返回:
        loss: float, 越大表示特征越难被预测
    """
    n_samples, n_features = pred_train.shape
    total_loss = 0.0

    for i in range(n_features):
        Zi = pred_train[:, i]
        Z_others = np.delete(pred_train, i, axis=1)

        # 使用非线性回归模型预测 Zi
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        model.fit(Z_others, Zi)
        pred = model.predict(Z_others)

        # R² 越小 → Zi 越不可预测
        r2 = r2_score(Zi, pred)
        # 由于我们希望 loss 越大越好
        loss_i = 1 - r2
        total_loss += loss_i

    # 平均每个特征的不可预测性
    loss = total_loss / n_features
    return loss


def nonpredictability_loss_lgb_r2(preds_train, n_estimators=50, random_state=0):
    """
    CPU-friendly LGBM 计算每列特征不可预测性 loss
    
    参数:
        preds_train: np.ndarray, shape [n_samples, n_features]
        sample_ratio: 抽取样本比例，用于加速
        subset_cols: 每列只用部分其他特征预测
        n_estimators: LGBM 树的数量
        max_depth: LGBM 树深度
    返回:
        loss: float, 越大表示特征越不可预测
    """
    n_samples, n_features = preds_train.shape
    total_loss = 0.0
    
    for i in range(n_features):
        Zi = preds_train[:, i]
        Z_others = np.delete(preds_train, i, axis=1)
        
        # 使用非线性回归模型预测 Zi
        model = LGBMRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1,  # 不输出日志
            n_jobs=-1    # 使用所有CPU核心
        )
        model.fit(Z_others, Zi)
        pred = model.predict(Z_others)
        
        r2 = r2_score(Zi, pred)
        loss_i = max(0.0, r2) 
        total_loss += loss_i

    # 平均不可预测性
    loss = total_loss / n_features
    return loss

def KraskovMI1_nats(x, y, k=1):
    N, dim = x.shape
    V = np.hstack([x, y])
    kdtree = scispa.KDTree(V)
    ei, _ = kdtree.query(V, k+1, p=np.inf)
    dM = ei[:, -1]
    nx = scispa.KDTree(x).query_ball_point(x, dM, p=np.inf, return_length=True)
    ny = scispa.KDTree(y).query_ball_point(y, dM, p=np.inf, return_length=True)
    ave = (psi(nx) + psi(ny)).mean()
    return psi(k) - ave + psi(N)

def pairwise_independence(pi_matrix, k=5):
    """
    Args:
        pi_matrix: shape (n_samples, n_pi+1)，最后一列是 Y
    """
    pi_matrix = np.asarray(pi_matrix)
    if pi_matrix.ndim != 2:
        raise ValueError("pi_matrix must be 2D")
    
    n_pi = pi_matrix.shape[1] - 1  # 去掉最后的 Y 列
    
    # 分离 Pi 和 Y
    pi_data = pi_matrix[:, :n_pi]    # 前 n_pi 列
    y_data = pi_matrix[:, -1:]        # 最后一列（Y）
    
    # 1. 计算 Pi 之间的冗余（两两 MI）
    redundancy_score = 0.0
    if n_pi >= 2:
        pairwise_mi = np.zeros((n_pi, n_pi))
        for i in range(n_pi):
            for j in range(i + 1, n_pi):
                mi = KraskovMI1_nats(
                    pi_data[:, i].reshape(-1, 1),
                    pi_data[:, j].reshape(-1, 1),
                    k
                )
                pairwise_mi[i, j] = mi
                pairwise_mi[j, i] = mi
        redundancy_score = np.mean(pairwise_mi[np.triu_indices_from(pairwise_mi, k=1)])
    # 2. 计算每个 Pi 与 Y 的互信息
    mi_with_y = np.zeros(n_pi)
    for i in range(n_pi):
        mi = KraskovMI1_nats(
            pi_data[:, i].reshape(-1, 1),
            y_data,
            k
        )
        mi_with_y[i] = mi
    
    # 3. 计算指标
    information_score = np.mean(mi_with_y)
    total_score = redundancy_score - information_score  # 最小化目标
    
    return total_score