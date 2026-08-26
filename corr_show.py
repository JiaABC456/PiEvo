from scipy.linalg import null_space
import numpy as np
import json
import pandas as pd
import os
import warnings
from data_Loss import rf_mse, lgb_mse
from utils import remove_bad_rows, visualize_full_corr_matrix
from generate_data import *

warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')

file_path = "./configs/Rayleigh/"
individuals = np.load(file_path + "final_results/individuals.npy")
c_masks = np.load(file_path + "final_results/c_masks.npy")

config_file= file_path + "config.json"

with open(config_file, 'r') as f:
    config = json.load(f)

c_num = config.get("C_NUM", 10)
population_size = config.get("POPULATION_SIZE", 50)
max_generations = config.get("MAX_GENERATIONS", 100)
dim_matrix = np.array(config.get("DIM_MATRIX"))
config_file = os.path.dirname(config_file)
data_path = os.path.join(config_file, config.get("DATA_PATH", "chf_public.csv"))
# 数据加载
X, y, variable_names = handle_rayleigh_data(data_path)
X_train, y_train = X[:int(X.shape[0]*0.8),:], y[:int(y.shape[0]*0.8)]
X_val, y_val = X[int(X.shape[0]*0.8):,:], y[int(y.shape[0]*0.8):]

D = dim_matrix[:, :-1]      # k x n
d = dim_matrix[:, -1].reshape(-1, 1)      # k

W_p, _, _, _ = np.linalg.lstsq(D, d, rcond=None)  # n x 1

N = null_space(D)   # shape: (n, r)


r = N.shape[1]
W_p_expanded_repeat = np.repeat(W_p, c_num, axis=1)

# 遍历所有的解
num_solutions = individuals.shape[0]
results = []

for solution_idx in range(num_solutions):
    c_vector = individuals[solution_idx].reshape(-1, 1)  # (c_num * r, 1)
    c_mask = c_masks[solution_idx].reshape(-1, 1)
    c_num_current = c_mask.shape[0]
    
    # 得到最后的x
    repeated_c_vector = np.repeat(c_mask, r, axis=0)
    masked_c_vector = c_vector * repeated_c_vector # c_num * r
    masked_c_vector = masked_c_vector.reshape(r, -1, order='F')
    x = W_p_expanded_repeat + N @ masked_c_vector   # n* c_num
    
    # 生成训练集预测
    preds = []
    for i in range(c_num_current):
        xi = x[:, i].reshape(-1, 1)  # n x 1
        pred_i = np.prod(X ** xi.T, axis=1)  # m x 1
        preds.append(pred_i)
    preds = np.column_stack(preds)  # m x c_num
    preds_train, y_train = remove_bad_rows(preds, y)
    preds_val, y_val = remove_bad_rows(preds[int(X.shape[0]*0.8):,:], y[int(y.shape[0]*0.8):])
    # 生成验证集预测
    
    preds_train_masked = preds_train * c_mask.T
    preds_val_masked = preds_val * c_mask.T
    # 训练 random forest 
    mse, r2, feature_importances = rf_mse(preds_train_masked, y_train, preds_val_masked, y_val)
    
    results.append({
        'Solution_Index': solution_idx,
        'R2': r2,
        'MSE': mse,
        'Feature_Importances': feature_importances
    })
    # 可视化相关矩阵
    corr_matrix = np.corrcoef(preds_train_masked.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 0)
    visualize_full_corr_matrix(corr_matrix, c_mask=c_mask,
                              save_path= file_path + f'pop_imgs/corr_matrix/corr_matrix_solution_{solution_idx}.png')
    print(f"Solution {solution_idx}: R2={r2:.6f}, MSE={mse:.6f}")
    print(f"Feature Importances: {feature_importances}")

# 打印汇总信息
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
r2_values = [res['R2'] for res in results]
print(f"Best R2: {max(r2_values):.6f} (Solution {np.argmax(r2_values)})")
print(f"Mean R2: {np.mean(r2_values):.6f}")
print(f"Min R2: {min(r2_values):.6f}")

# 保存结果到CSV
results_df = pd.DataFrame(results)
# 将Feature_Importances转换为字符串格式以便保存
results_df['Feature_Importances'] = results_df['Feature_Importances'].apply(
    lambda x: ','.join(map(str, x)) if isinstance(x, np.ndarray) else str(x)
)
output_path = file_path + "final_results/evaluation_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

