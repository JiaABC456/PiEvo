import numpy as np
import random
import pandas as pd
import json
import os
from scipy.linalg import null_space
from utils import *
from init import initialize2, initialize3
from evaluate import evaluate, evaluate_population
from tqdm import tqdm
from generate_data import *

FUNCTION_MAP = {
    "handle_CHF_data": handle_CHF_data,
    "handle_velocity_trans_data": handle_velocity_trans_data,
    "handle_rayleigh_data": handle_rayleigh_data,
    "handle_Keyhole_data": handle_Keyhole_data,
    "handle_Wall_model": handle_Wall_model,
    "handle_MHD_data": handle_MHD_data,
}


class DEA:
    def __init__(self, config_file="config.json", data: tuple = None):

        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.config_file = os.path.dirname(config_file)

        self.seed = self.config.get("SEED", 42)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.c_num = self.config.get("C_NUM", 10)
        self.population_size = self.config.get("POPULATION_SIZE", 50)
        self.max_generations = self.config.get("MAX_GENERATIONS", 100)
        self.dim_matrix = np.array(self.config.get("DIM_MATRIX"))
        self.data_path = os.path.join(self.config_file, self.config.get("DATA_PATH", "chf_public.csv"))

        # 数据加载
        if data is None:
            func_name = self.config.get("FUNCTION", "handle_CHF_data")
            self.handle_data_function = FUNCTION_MAP.get(func_name)
            self.X, self.y, self.variable_names = self.handle_data_function(self.data_path)
        else:
            self.X, self.y, self.variable_names = data
        
        # 数据集划分
        _, _, _, _, _, _, self.idx_train, self.idx_val, self.idx_test = sklearn_split_method_with_indices(self.X, self.y)
        
        # 量纲分析
        self.D = self.dim_matrix[:, :-1]      # k x n
        self.d = self.dim_matrix[:, -1].reshape(-1, 1)      # k
        self.W_p, _, _, _ = np.linalg.lstsq(self.D, self.d, rcond=None)  # n x 1

        self.N = null_space(self.D)   # shape: (n, r)
        self.r = self.N.shape[1]
        self.W_p_expanded_repeat = np.repeat(self.W_p, self.c_num, axis=1)

        # 种群
        self.individuals = None
        self.c_masks = None
        self.results = []
        self.FrontNo = None
        self.CrowdDis = None
        self.feature_importances = []
        self.correlations = []
        self.current_generation = 0
        self.lower_bound = self.config.get("lower_bound", -5)
        self.upper_bound = self.config.get("upper_bound", 5)

    def initialize_population(self):
        """
        初始化种群  第一个初始化就结束了  进行第二个初始化操作  进行第三个初始化。
        """
        # individuals1, c_masks1 = initialize1(self.data_path,self.handle_data_function,self.c_num, self.population_size, self.idx_train, self.idx_test)
        individuals2, c_masks2 = initialize2(self.c_num, self.population_size, self.X, self.y, self.W_p, self.N, lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        individuals3, c_masks3 = initialize3(self.c_num, self.population_size, self.r, lower_bound=self.lower_bound, upper_bound=self.upper_bound)
        # 合并三种初始化方法的个体和掩码
        individuals = np.vstack((individuals2, individuals3))
        # individuals = np.hstack((individuals, individuals1))
        c_masks = np.vstack((c_masks2, c_masks3))
        # c_masks = np.hstack((c_masks, c_masks1))

        return individuals, c_masks

    def run(self):
        """
        运行进化算法
        """
        self.individuals, self.c_masks = self.initialize_population()
        # 种群评价
        for idx in tqdm(range(self.individuals.shape[0]), desc="Evaluating population"):
            individual = self.individuals[idx]
            c_mask = self.c_masks[idx]
            mse, feature_importances, orth_loss, correlation = evaluate(individual, c_mask, W_p_expanded_repeat = self.W_p_expanded_repeat, N = self.N,
                                        idx_train = self.idx_train, idx_test = self.idx_test, X = self.X, y = self.y, r = self.r)
            self.results.append((mse, orth_loss))
            self.feature_importances.append(feature_importances)
            self.correlations.append(correlation)
        self.results = np.array(self.results)  # shape: (pop_size, 2)
        self.feature_importances = np.array(self.feature_importances)
        self.correlations = np.array(self.correlations)

        self.results, self.individuals, self.c_masks, self.feature_importances, self.correlations, self.FrontNo, self.CrowdDis = self.Selection(self.results, self.individuals, self.c_masks, 
                                                                                                                             self.feature_importances, self.correlations
                                                                                                                            ,self.population_size)
        # plot_population(self.results, self.c_masks, save_path = os.path.join(self.config_file,"pop_imgs/initial_population.png"))
        # 进化过程
        for self.current_generation in tqdm(range(self.max_generations), desc="Evolving"):
            MatingPool = tournament_selection(2, 2*self.population_size, self.FrontNo, -self.CrowdDis)
            # 生成子代
            self.Off_individuals, self.Off_c_masks = self.Operator1(self.individuals[MatingPool], self.c_masks[MatingPool], self.feature_importances[MatingPool], self.correlations[MatingPool])
            # 种群评价
            self.Off_results, self.Off_feature_importances, self.Off_correlations = evaluate_population(self.Off_individuals, self.Off_c_masks, W_p_expanded_repeat = self.W_p_expanded_repeat, N = self.N,
                                        idx_train = self.idx_train, idx_test = self.idx_test, X = self.X, y = self.y, r = self.r)
            self.results, self.individuals, self.c_masks, self.feature_importances, self.correlations, self.FrontNo, self.CrowdDis = self.Selection(np.vstack((self.results, self.Off_results)), 
                                                                                                       np.vstack((self.individuals, self.Off_individuals)), 
                                                                                                       np.vstack((self.c_masks, self.Off_c_masks)), 
                                                                                                       np.vstack((self.feature_importances, self.Off_feature_importances)),
                                                                                                       np.vstack((self.correlations, self.Off_correlations)),
                                                                                                       self.population_size)
            plot_population(self.results, self.c_masks,save_path = os.path.join(self.config_file,"pop_imgs/current_population.png"))
        # # 保存self.results, self.individuals, self.c_masks
        # final_dir = os.path.join(self.config_file, "final_results")
        # if not os.path.exists(final_dir):
        #     os.makedirs(final_dir, exist_ok=True)
        # np.save(os.path.join(final_dir, "results.npy"), self.results)
        # np.save(os.path.join(final_dir, "individuals.npy"), self.individuals)
        # np.save(os.path.join(final_dir, "c_masks.npy"), self.c_masks)
        # # 在验证集上评估最终种群
        # self.val_results, self.val_feature_importances, self.val_r2 = evaluate_population(self.individuals, self.c_masks, W_p_expanded_repeat = self.W_p_expanded_repeat, N = self.N,
        #                                 idx_train = self.idx_train, idx_test = self.idx_val, X = self.X, y = self.y, r = self.r)
        # plot_population(self.results, self.c_masks, save_path = os.path.join(self.config_file,"pop_imgs/final_population.png"))
        # plot_population(self.val_results, self.c_masks, save_path = os.path.join(self.config_file,"pop_imgs/val_final_population.png"))
        # # 保存解的分析结果
        # if self.config.get("SAVE_SOLUTION_ANALYSIS", True):
        #     self.save_solutions_to_csv(self.individuals, self.c_masks, self.results, save_path=os.path.join(final_dir, "solutions_analysis.csv"))
        #     self.save_pareto_solutions(self.individuals, self.c_masks, self.results, save_path=os.path.join(final_dir, "pareto_front_solutions.csv"))
        #     self.visualize_all_solutions_pi(self.individuals, self.c_masks)
    def Crossover(self, individuals, c_masks, feature_importances):
        """
        交叉操作
  
        现在你也是优化c和c_mask, 对于c的优化要结合表格数据特征, 对于c_mask的优化更要结合表格特征。
        这个是变异操作，而且你说的高斯过程回归没有涉及到表格这个东西，任何一个算法都会这样做？
        
        包括c_mask的交叉和c的交叉 
        
        把前c_num重要的放在一起 -> 会导致拟合mse减小吧! 
        一个原因是 当c相似时不能保证无量纲数是相似的。 所以c的正交性是错误的。
        那么这些重要性的在一起交叉 会导致冗余吗？我想会。但是还要试一试。

        """
        pass

    def Mutation(self, individuals, c_masks, features_importances):
        """
        变异操作
        我也依然是有百分之多少可以打开c_mask, 多少关闭c_mask
        我还必须根据返回的特征重要性来关闭一些c_mask

        c的变异 第一个就是在原有的基础上加减一个小的值

        一个无量纲数在一个系统中的特征重要性很低，在另一个特征重要性中依然也很低。
        难道是0的就一直是0吗?  
        """
        N, D = individuals.shape
        Parent1Mask = c_masks[:N//2, :]
        Parent2Mask = c_masks[N//2:, :]

        indices = list(zip(*np.where(c_masks == 1)))
        individuals = individuals.reshape(-1, self.c_num, self.r)
        # 首先, 把重要性小的1置为0。
        # 把所有之前为1的c和对应的feature,放在一个容器中，进行训练
        # 预测0->1的概率，并改变0->1 (这个只针对之前为0的)  
        # 单个c到特征重要性的映射。
        # 必须有一个机制让信息聚集在少数的无量纲组上  能聚集就聚集
        pass


    def Selection(self, PopObj, Dec, Mask, Features_importance, Correlations, N):

        # ----------------- 删除重复解 -----------------
        _, unique_idx = np.unique(PopObj, axis=0, return_index=True)
        PopObj = PopObj[unique_idx]
        Dec = Dec[unique_idx]
        Mask = Mask[unique_idx]
        Features_importance = Features_importance[unique_idx]
        Correlations = Correlations[unique_idx]

        N = min(N, len(PopObj))

        # ----------------- 非支配排序 -----------------
        rank, MaxFNo = fast_non_dominated_sort(PopObj, N)
        # print("MaxFNo:", MaxFNo)
        # ----------------- 计算拥挤度 -----------------
        CrowdDis = crowding_distance(PopObj, rank)

        # ----------------- 选出前 MaxFNo-1 个前沿 -----------------
        Next = rank < MaxFNo

        # ----------------- 处理最后一个前沿 -----------------
        last_idx = np.where(rank == MaxFNo)[0]
        if len(last_idx) > 0:
            sorted_idx = last_idx[np.argsort(-CrowdDis[last_idx])]  # 拥挤度降序
            remaining = N - np.sum(Next)
            Next[sorted_idx[:remaining]] = True


        # ----------------- 生成下一代种群 -----------------
        SelectedPopObj = PopObj[Next]
        SelectedDec = Dec[Next]
        SelectedMask = Mask[Next]
        SelectedFeatureImportance = Features_importance[Next]
        SelectedCorrelations = Correlations[Next]
        FrontNo = rank[Next]
        CrowdDis = CrowdDis[Next]

        return SelectedPopObj, SelectedDec, SelectedMask, SelectedFeatureImportance, SelectedCorrelations, FrontNo, CrowdDis
    
    
    def Operator1(self, individuals, c_masks, feature_importances, correlations):
        """
        Π-level crossover: c 和 c_mask 联合交叉

        对每一对父代：
        - 合并 Parent1 和 Parent2 的 Π
        - 按 feature importance 排序
        - 选前 c_num 个 Π
        - 对应的 c 向量整体拷贝到 offspring
        """
        N, D = individuals.shape
        Parent1Mask = c_masks[:N//2, :]

        OffMask = Parent1Mask.copy()
        for i in range(N//2):
            if np.random.rand() < 0.5:
                # 1 -> 0 的变异
                indices = np.where(OffMask[i] == 1)[0]
                local_indices = TS(feature_importances[i][indices])
                global_indices = indices[local_indices]  # binary tournament selection
                OffMask[i, global_indices] = 0
            else:
                # 0 -> 1 的变异
                indices = np.where(OffMask[i] == 0)[0]
                local_indices = TS(-feature_importances[i][indices])
                global_indices = indices[local_indices]
                OffMask[i, global_indices] = 1

            if np.random.rand() < 0.5:
                # 1 -> 0 的变异
                indices = np.where(OffMask[i] == 1)[0]
                local_indices = TS(-correlations[i][indices])  # binary tournament selection
                global_indices = indices[local_indices]
                OffMask[i, global_indices] = 0
            else:
                # 0 -> 1 的变异
                indices = np.where(OffMask[i] == 0)[0]
                local_indices = TS(correlations[i][indices])  # binary tournament selection
                global_indices = indices[local_indices]
                OffMask[i, global_indices] = 1

        # 去除全为0的掩码以及对应的解 这些都完蛋了。
        OffDec = GAreal(individuals[:N//2, :], individuals[N//2:, :], feature_importances,
                        np.repeat(self.lower_bound, self.individuals.shape[1]), 
                        np.repeat(self.upper_bound, self.individuals.shape[1]), self.c_num, self.r,
                        proC=0.9, disC=20, proM=0.1, disM=20)
        
        # 去除c_mask全为0的个体
        valid_mask = ~np.all(OffMask == 0, axis=1)
        OffDec = OffDec[valid_mask]
        OffMask = OffMask[valid_mask]
        
        # 去除重复的个体（基于OffDec和OffMask的组合）
        if len(OffDec) > 0:
            combined = np.hstack([OffDec, OffMask])
            _, unique_indices = np.unique(combined, axis=0, return_index=True)
            unique_indices = np.sort(unique_indices)  # 保持原始顺序
            OffDec = OffDec[unique_indices]
            OffMask = OffMask[unique_indices]
        
        return OffDec, OffMask
    

    # def Operator2(self, individuals, c_masks, feature_importances, correlations):
    #     """
    #     全新的、知识引导的遗传算子。
    #     - 交叉: 特征重要性引导的精英重组 (IGX)。
    #     - 变异: 
    #         - c_mask: 冗余感知的剪枝/生长。
    #         - c: 重要性自适应的参数微调。
    #     """
    #     N_parents, D = individuals.shape
    #     N = N_parents // 2  # 我们将生成 N 个子代
        
    #     # 从 config 中获取变异的概率和强度参数
    #     prob_mask_prune = self.config.get("PROB_MASK_PRUNE", 0.1)  # 剪枝概率
    #     prob_mask_grow = self.config.get("PROB_MASK_GROW", 0.05)   # 生长概率
    #     prob_c_tune = self.config.get("PROB_C_TUNE", 0.8)        # c向量变异概率
        
    #     # 分割父代
    #     Parent1_dec = individuals[:N, :]
    #     Parent2_dec = individuals[N:, :]
    #     Parent1_mask = c_masks[:N, :]
    #     Parent2_mask = c_masks[N:, :]
    #     Parent1_imp = feature_importances[:N, :]
    #     Parent2_imp = feature_importances[N:, :]

    #     # 准备存储子代的数组
    #     Offspring_dec = np.zeros_like(Parent1_dec)
    #     Offspring_mask = np.zeros_like(Parent1_mask)

    #     # --- 1. 精英重组交叉 (IGX) ---
    #     for i in range(N):
    #         # 1.1 合并精英池
    #         # 找出父代A和B中所有激活的Pi组的索引
    #         active_indices_A = np.where(Parent1_mask[i] == 1)[0]
    #         active_indices_B = np.where(Parent2_mask[i] == 1)[0]
            
    #         # 将所有激活的Pi组信息（c向量片段，重要性）放入精英池
    #         elite_pool = []
    #         for idx in active_indices_A:
    #             c_segment = Parent1_dec[i, idx * self.r : (idx + 1) * self.r]
    #             imp = Parent1_imp[i, idx]
    #             # (重要性, c片段, 原始索引)
    #             elite_pool.append((-imp, c_segment, idx)) # 用负重要性，因为后面要按升序排
                
    #         for idx in active_indices_B:
    #             # 避免重复添加
    #             if not any(item[2] == idx and np.array_equal(item[1], Parent2_dec[i, idx * self.r : (idx + 1) * self.r]) for item in elite_pool):
    #                 c_segment = Parent2_dec[i, idx * self.r : (idx + 1) * self.r]
    #                 imp = Parent2_imp[i, idx]
    #                 elite_pool.append((-imp, c_segment, idx))

    #         # 1.2 择优录取
    #         # 按重要性从高到低排序
    #         elite_pool.sort(key=lambda x: x[0])
            
    #         # 选择前 c_num 个最优秀的Pi组
    #         num_to_select = min(self.c_num, len(elite_pool))
    #         selected_elites = elite_pool[:num_to_select]

    #         # 1.3 构建子代
    #         if not selected_elites: continue # 如果父母都没有激活的Pi，子代为空

    #         for j, elite in enumerate(selected_elites):
    #             _, c_segment, original_idx = elite
    #             # 新的Pi组放在子代的前j个位置
    #             Offspring_mask[i, j] = 1
    #             Offspring_dec[i, j * self.r : (j + 1) * self.r] = c_segment

    #     # --- 2. 分层变异 ---
    #     for i in range(N):
    #         # 2.1 c_mask 变异 (冗余感知的剪枝/生长)
    #         if np.random.rand() < prob_mask_prune:
    #             active_indices = np.where(Offspring_mask[i] == 1)[0]
    #             if len(active_indices) > 1: # 至少要保留一个
    #                 # 找到最差的Pi组：冗余度最高 或 重要性最低
    #                 # 这里我们用相关性/冗余度作为例子
    #                 corrs_of_active = correlations[i, active_indices]
    #                 worst_pi_local_idx = np.argmax(corrs_of_active) # 冗余度最高的
    #                 worst_pi_global_idx = active_indices[worst_pi_local_idx]
    #                 Offspring_mask[i, worst_pi_global_idx] = 0 # 剪枝！
            
    #         if np.random.rand() < prob_mask_grow:
    #             inactive_indices = np.where(Offspring_mask[i] == 0)[0]
    #             if len(inactive_indices) > 0:
    #                 # 随机选择一个未激活的位置来生长
    #                 grow_idx = np.random.choice(inactive_indices)
    #                 Offspring_mask[i, grow_idx] = 1
    #                 # 同时需要初始化对应的c向量片段
    #                 Offspring_dec[i, grow_idx * self.r : (grow_idx + 1) * self.r] = \
    #                     np.random.uniform(self.lower_bound, self.upper_bound, self.r)

    #         # 2.2 c 变异 (重要性自适应的参数微调)
    #         if np.random.rand() < prob_c_tune:
    #             active_indices = np.where(Offspring_mask[i] == 1)[0]
    #             if len(active_indices) > 0:
    #                 imps_of_active = feature_importances[i, active_indices]
    #                 # 归一化重要性，作为变异强度的参考
    #                 norm_imps = (imps_of_active - np.min(imps_of_active)) / (np.max(imps_of_active) - np.min(imps_of_active) + 1e-9)

    #                 for j, idx in enumerate(active_indices):
    #                     # 变异强度与重要性成反比
    #                     # 重要性越高 (norm_imps接近1), 扰动越小
    #                     # 重要性越低 (norm_imps接近0), 扰动越大
    #                     mutation_strength = 0.5 * (1 - norm_imps[j]) # 强度范围 [0, 0.5]
                        
    #                     c_segment = Offspring_dec[i, idx * self.r : (idx + 1) * self.r]
    #                     noise = np.random.normal(0, mutation_strength, self.r)
                        
    #                     mutated_c_segment = c_segment + noise
                        
    #                     # 确保不出界
    #                     Offspring_dec[i, idx * self.r : (idx + 1) * self.r] = \
    #                         np.clip(mutated_c_segment, self.lower_bound, self.upper_bound)

    #     # --- 3. 清理无效个体 ---
    #     # 去除 c_mask 全为0的个体
    #     valid_mask_indices = ~np.all(Offspring_mask == 0, axis=1)
    #     if not np.any(valid_mask_indices):
    #         return np.array([]), np.array([]) # 如果所有子代都无效，返回空

    #     Offspring_dec = Offspring_dec[valid_mask_indices]
    #     Offspring_mask = Offspring_mask[valid_mask_indices]

    #     # 去除重复的个体 (基于 dec 和 mask 的组合)
    #     if len(Offspring_dec) > 0:
    #         combined = np.hstack([Offspring_dec, Offspring_mask])
    #         _, unique_indices = np.unique(combined, axis=0, return_index=True)
    #         Offspring_dec = Offspring_dec[unique_indices]
    #         Offspring_mask = Offspring_mask[unique_indices]

    #     return Offspring_dec, Offspring_mask

    def info(self):
        """
        打印配置信息
        """
        print("DEA Configuration:")
        print(f"C_NUM: {self.c_num}")
        print(f"POPULATION_SIZE: {self.population_size}")
        print(f"MAX_GENERATIONS: {self.max_generations}")
        print(f"DATA_PATH: {self.data_path}")
    

    ##### 分析和保存解的函数 #####
    def save_solutions_to_csv(self, individuals, c_masks, results, save_path="solutions_analysis.csv"):
        """
        将种群中所有解的无量纲数组合和目标函数值保存到CSV文件
        
        Parameters
        ----------
        individuals : ndarray, shape (population_size, c_num * r)
            种群中的所有个体
        c_masks : ndarray, shape (population_size, c_num)
            每个个体的掩码矩阵
        results : ndarray, shape (population_size, 2)
            每个个体的目标函数值 [prediction_error, orthogonality_loss]
        save_path : str
            保存CSV文件的路径
        """
        # 确保保存目录存在
        save_dir = Path(save_path).parent
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备存储所有数据的列表
        all_data = []
        
        print(f"开始分析 {len(individuals)} 个解...")
        
        for i in range(len(individuals)):
            individual = individuals[i]
            c_mask = c_masks[i]
            objective_values = results[i]  # [目标1, 目标2]
            
            # 解码得到无量纲数的指数矩阵
            c_vector = individual.reshape(-1, 1)
            c_mask_reshaped = c_mask.reshape(-1, 1)
            c_num = c_mask_reshaped.shape[0]
            
            # 计算无量纲数矩阵
            repeated_c_vector = np.repeat(c_mask_reshaped, self.r, axis=0)
            masked_c_vector = c_vector * repeated_c_vector
            masked_c_vector = masked_c_vector.reshape(self.r, -1, order='F')
            x_matrix = self.W_p_expanded_repeat + self.N @ masked_c_vector
            
            # 生成无量纲数标签
            labels = create_labels(x_matrix.T, self.variable_names)
            
            # 提取激活的无量纲数
            active_pi_labels = []
            active_pi_count = 0
            
            for j, label in enumerate(labels):
                if c_mask[j] == 1:  # 只记录激活的无量纲数
                    active_pi_count += 1
                    active_pi_labels.append(label.strip('$'))  # 去除LaTeX的$符号
            
            # 构建该解的数据行
            solution_data = {
                'solution_id': i,
                'num_active_pi': active_pi_count,
                'objective_1': objective_values[0],  # 预测误差
                'objective_2': objective_values[1],  # 正交性损失
                'active_pi_count': active_pi_count
            }
            
            # 添加每个无量纲数的信息（最多支持10个）
            for pi_idx in range(10):  # 假设最多10个无量纲数
                if pi_idx < len(active_pi_labels):
                    solution_data[f'pi_{pi_idx+1}'] = active_pi_labels[pi_idx]
                else:
                    solution_data[f'pi_{pi_idx+1}'] = ''
            
            # 添加掩码信息
            solution_data['c_mask'] = ''.join(str(int(m)) for m in c_mask)
            
            all_data.append(solution_data)
        
        # 创建DataFrame
        df = pd.DataFrame(all_data)
        
        # 重新排列列的顺序，让重要信息在前
        columns_order = ['solution_id', 'num_active_pi', 'objective_1', 'objective_2', 'c_mask', 'active_pi_count']
        pi_columns = [col for col in df.columns if col.startswith('pi_')]
        other_columns = [col for col in df.columns if col not in columns_order + pi_columns]
        
        df = df[columns_order + pi_columns + other_columns]
        
        # 保存到CSV
        df.to_csv(save_path, index=False, encoding='utf-8')
        
        print(f"解分析已保存到: {save_path}")
        print(f"共保存 {len(df)} 个解的信息")
        print(f"激活无量纲数统计: 平均 {df['num_active_pi'].mean():.1f} 个, "
            f"最少 {df['num_active_pi'].min()} 个, 最多 {df['num_active_pi'].max()} 个")
        
        return df

    def save_pareto_solutions(self, individuals, c_masks, results, save_path="pareto_front_solutions.csv"):
        """
        专门保存Pareto前沿上的解
        
        Parameters
        ----------
        参数同上
        save_path : str
            保存Pareto解的CSV文件路径
        """
        # 确保保存目录存在
        save_dir = Path(save_path).parent
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 非支配排序
        front_no, _ = fast_non_dominated_sort(results, individuals.shape[0])
        pareto_indices = np.where(front_no == 1)[0]  # 第一前沿是Pareto前沿
        
        print(f"找到 {len(pareto_indices)} 个Pareto最优解")
        
        if len(pareto_indices) == 0:
            print("警告: 未找到Pareto最优解")
            return None
        
        # 提取Pareto解
        pareto_individuals = individuals[pareto_indices]
        pareto_c_masks = c_masks[pareto_indices]
        pareto_results = results[pareto_indices]
        
        # 保存Pareto解
        pareto_df = self.save_solutions_to_csv(
            pareto_individuals, pareto_c_masks, pareto_results, save_path
        )
        
        # 添加Pareto前沿排名信息
        pareto_df['pareto_rank'] = 1
        
        # 按目标1排序，便于分析
        pareto_df_sorted = pareto_df.sort_values('objective_1')
        pareto_df_sorted.to_csv(save_path, index=False, encoding='utf-8')
        
        print(f"Pareto前沿解已保存到: {save_path}")
        
        return pareto_df_sorted

    def visualize_all_solutions_pi(self, individuals, c_masks, save_dir=None):
        """
        可视化所有解的无量纲数与因变量之间的二维散点图
        
        Parameters
        ----------
        individuals : ndarray, shape (population_size, c_num * r)
            所有个体
        c_masks : ndarray, shape (population_size, c_num)
            所有个体的掩码矩阵
        save_dir : str, optional
            保存图片的文件夹路径，如果为None则保存到默认路径
        """
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        def has_chinese(text):
            text = str(text)
            return any('\u4e00' <= ch <= '\u9fff' for ch in text)

        # Global defaults: English text uses Times New Roman.
        plt.rcParams['font.family'] = ['Times New Roman', 'serif']
        plt.rcParams['axes.unicode_minus'] = False

        zh_font = font_manager.FontProperties(
            family=['KaiTi', 'STKaiti', 'KaiTi_GB2312', 'SimKai']
        )
        en_font = font_manager.FontProperties(
            family=['Times New Roman', 'serif']
        )

        def pick_font(text):
            return zh_font if has_chinese(text) else en_font

        def to_math_pi(text):
            # Replace unicode Pi with mathtext Pi for proper mathematical rendering.
            return str(text).replace('Π', r'$\Pi$')
        
        # 设置默认保存路径
        if save_dir is None:
            save_dir = os.path.join(self.config_file, "pop_imgs", "pi_vs_target")
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始可视化 {len(individuals)} 个解...")
        
        # 遍历所有个体
        for sol_idx in tqdm(range(len(individuals)), desc="Generating plots"):
            individual = individuals[sol_idx]
            c_mask = c_masks[sol_idx]
            
            # 解码得到无量纲数的指数矩阵
            c_vector = individual.reshape(-1, 1)
            c_mask_reshaped = c_mask.reshape(-1, 1)
            c_num = c_mask_reshaped.shape[0]
            
            repeated_c_vector = np.repeat(c_mask_reshaped, self.r, axis=0)
            masked_c_vector = c_vector * repeated_c_vector
            masked_c_vector = masked_c_vector.reshape(self.r, -1, order='F')
            x = self.W_p_expanded_repeat + self.N @ masked_c_vector  # n x c_num
            
            # 计算每个无量纲数的值
            pi_values = []
            pi_labels = []
            active_indices = np.where(c_mask == 1)[0]
            
            for i in active_indices:
                xi = x[:, i].reshape(-1, 1)  # n x 1
                pi_i = np.prod(self.X ** xi.T, axis=1)  # m x 1
                pi_values.append(pi_i)
            
            if len(pi_values) == 0:
                print(f"Solution {sol_idx}: 没有激活的无量纲数，跳过")
                continue
            
            # 生成标签
            labels = create_labels(x.T, self.variable_names)
            for i in active_indices:
                pi_labels.append(labels[i])
            
            # 计算子图布局
            n_plots = len(pi_values)
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            # 创建画布
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_plots == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            # 绘制每个无量纲数与因变量的关系
            for idx, (pi_val, label) in enumerate(zip(pi_values, pi_labels)):
                ax = axes[idx]

                pi_val_clean = pi_val
                y_clean = self.y
                
                ax.scatter(pi_val_clean, y_clean, alpha=0.6, s=20)
                x_label = rf'$\Pi_{{find}}=$' + ' ' + to_math_pi(label)
                ax.set_xlabel(x_label, fontsize=10, fontproperties=pick_font(label))
                y_label = rf'$\Pi_{{true}}$'
                # title_text = rf'$\Pi_{{{active_indices[idx]+1}}}$ vs $\Pi$'
                ax.set_ylabel(y_label, fontsize=10, fontproperties=pick_font(y_label))
                # ax.set_title(title_text, fontsize=11, fontproperties=pick_font(title_text))
                ax.grid(True, alpha=0.3)

                # Keep numeric ticks in Times New Roman for consistent English styling.
                for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                    tick_label.set_fontproperties(en_font)
            
            # 隐藏多余的子图
            for idx in range(n_plots, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            # 保存图片
            save_path = os.path.join(save_dir, f"solution_{sol_idx:03d}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"所有无量纲数可视化已保存到: {save_dir}")



if __name__ == "__main__":
    dea = DEA(config_file="./configs/Rayleigh/config.json")
    dea.run()