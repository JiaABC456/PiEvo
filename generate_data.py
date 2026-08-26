
import numpy as np
from scipy.special import erf
import os
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io
def velocity_profile(y, nu, U, t):
    return U * (1 - erf(y / (2 * np.sqrt(nu * t))))

def generate_rayleigh_data():
    """完全按照您提供的采样方式生成数据"""
    # 您的采样参数
    y_vals = np.linspace(0, 1, 20)
    t_vals = np.linspace(5, 10, 20)
    U = np.random.uniform(0.5, 1.0, 5)
    nu = np.random.uniform(1e-3, 1e-2, 5)
    
    # 生成数据
    u_array = []
    params = []
    
    for u0, n0 in zip(U, nu):
        for y in y_vals:
            for t in t_vals:
                u_array.append(velocity_profile(y, n0, u0, t))
                params.append([y, n0, u0, t])
    
    u_array = np.array(u_array).reshape(-1, 1)
    params = np.array(params)
    u_array = u_array / params[:, 2].reshape(-1,1)
    # X: [y, nu, U, t], y: u
    X = params
    y = u_array.flatten()
    
    return X, y


def save_rayleigh_data_to_csv(X, y, config_dir="configs/V"):
    """
    直接将X和y保存到CSV文件
    
    Args:
        X: 特征数组 (n_samples, 4)
        y: 目标变量数组 (n_samples,)
        config_dir: 保存目录
    """
    # 确保目录存在
    output_dir = Path(config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=['y', 'nu', 'U', 't'])
    df['u'] = y  # 直接添加目标列
    
    # 保存到CSV
    csv_path = output_dir / "rayleigh_data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path

def handle_CHF_data(data_path):
    """
    处理CHF数据集的函数示例
    """
    # 这里可以添加处理CHF数据集的代码
    # 数据加载
    dataset = pd.read_csv("./configs/CHF/chf_public.csv")
    dataset = dataset.iloc[:, 2:-1]
    # 在倒数第二列插入重力加速度列（常数）
    g_value = 9.81
    dataset.insert(len(dataset.columns) - 1, 'g', g_value)

    # 自变量 X
    X = dataset.iloc[1:10000, :-1].values.astype(float)  # 所有列除了最后一列
    # 因变量 y
    y = dataset.iloc[1:10000, -1].values.astype(float)  # 最后一列
    
    all_variables = dataset.columns.tolist()
    feature_variables = all_variables[:-1]

    return X, y, feature_variables

def handle_rayleigh_data(data_path):
    """
    处理Rayleigh数据集的函数示例
    """
    # 这里可以添加处理Rayleigh数据集的代码
    dataset = pd.read_csv("./configs/Rayleigh/rayleigh_data.csv")
    # 自变量 X
    X = dataset.iloc[:, :-1].values.astype(float)  # 所有列除了最后一列
    # 因变量 y
    y = dataset.iloc[:, -1].values.astype(float)  # 最后一列

    all_variables = dataset.columns.tolist()
    feature_variables = all_variables[:-1]

    return X, y, feature_variables

def handle_velocity_trans_data(data_path):
    """
    处理Velocity Transformation数据集的函数示例
    """
    mat_data  = scipy.io.loadmat('./configs/Velocity_transformation/mat/inv_transf.mat')
    X         = mat_data['input'] # Load the .mat file, [Y1,Rho1,Mu1,Rhow,Muw,Utau]
    Y         = mat_data['output'] # The output is u/utau

    all_variables = ['Y1', 'Rho1', 'Mu1', 'Rhow', 'Muw', 'Utau', 'u_utau']

    return X, Y.flatten(), all_variables[:-1]

def handle_Wall_model(data_path):
    """
    处理Wall Model数据集的函数示例
    """
    # Load dataset
    dataset = np.loadtxt('./configs/Wall_model/dataset-GS1-GS10.txt').T
    rows_with_nan = np.isnan(dataset).any(axis=1)
    data = dataset[~rows_with_nan, :]
    Tw = 0.6756
    data[:,0] = data[:,0]-data[:,23]
    data[18460:, 20] += 59  # Adjusting for 0-based indexing in Python
    idy = (data[:, 0] > 0.02) & (data[:, 0] < 0.15)
    data = data[idy, :]
    idT = (data[:, 2] - Tw > 0)
    data = data[idT, :]
    condition = (data[:, 21] == 4) & (data[:, 22] == 4000)

    data = data[~condition]

    input_data = np.column_stack((data[:, 0], data[:, 1], data[:, 3], data[:, 2] , np.full_like(data[:,1], Tw), data[:, 4],
                                data[:, 7], data[:, 6], data[:, 11], data[:, 12], data[:, 15]))
    output_tauw = data[:, 18] * data[:, 0] / data[:, 4] / data[:, 1]
    output_qw = data[:, 19] * data[:, 0] / (data[:, 2] - Tw) / data[:, 7]
    index = data[:, 20]

    all_variables = ['y', 'u', 'ρ', 'T', 'T_w', 'μ', 'κ', 'c_p', 'k_rms', 'R_a', 'ES']


    return input_data, output_qw.flatten(), all_variables

def handle_MHD_data(data_path):
    """
    处理MHD数据集的函数示例
    """
    ## Load the dataset, handling invalid entries
    data = np.genfromtxt(
        "./configs/MHD/MHD_Generator_Data_Uavg.csv",
        delimiter=',',
        skip_header=1,
        usecols=(1,2,3,4,5,6),
        dtype=float,
        invalid_raise=False  # Ignore rows with invalid data
    )

    Xu = data[:, :5]
    u  = data[:, 5]
    mu    = (Xu[:,0])
    rho   =  (Xu[:,1])
    dp_dx =  (Xu[:,2])
    eta   =  (Xu[:,3])
    B     =  (Xu[:,4])
    l     = np.ones(mu.shape)

    X         = np.column_stack((l,mu,rho,eta,dp_dx,B))
    Y         = u*rho*l/mu

    all_variables = ['l','\\mu', '\\rho','\\eta','dp_dx','B']

    return X, Y.flatten(), all_variables

def handle_Keyhole_data(data_path):
    """
    处理Keyhole数据集的函数示例
    """
    df = pd.read_csv('./configs/Keyhole/dataset_keyhole.csv')

    output_list = ['e*']
    all_variables = ['etaP', 'Vs', 'rho', 'cp', 'Tl-T0', 'r0', 'alpha']
    X = np.array(df[all_variables])
    Y = np.array(df[output_list])

    return X, Y.flatten(), all_variables

if __name__ == "__main__":
    # 生成数据
    # X, y = generate_rayleigh_data()
    
    # # 保存数据到CSV
    # csv_path = save_rayleigh_data_to_csv(X, y, config_dir="configs/V")
    # print(f"Rayleigh data saved to {csv_path}")

    handle_Wall_model("dwadaw")