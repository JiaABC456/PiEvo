
# 初始化方法 ： 找到一个c，拿进化算法优化，让mse最小化，得到的c作为初始解之一。
# 其他的c就在周围进行初始化？
# 定量还有定方向
# c_mask 0-1只有对应的c为1。其他都为0。
import os
import json
import numpy as np
from utils import sign_match_probability
from matplotlib import pyplot as plt
# 和优化的变量有关系  应该把pi的正交性也加进去，要不然其他的解肯定完败这个初始化方法。 也就是正交性惩罚项和数据损失。复杂度损失由c_mask控制。
# 最后，使用PySR来得到最终的表达式形式？ PySR来优化无量纲数和q之间的关系。

# ga = GA()

# ga.run()
# ga.visualization()
# ga.save_results()


# Sign-aware Initialization
def initialize2(c_num, n, X, y, W_p, N, lower_bound=-5, upper_bound=5):

    y = y.flatten()
    m, n_features = X.shape
    sign_vec = np.zeros(n_features)

    for i in range(n_features):
        xi = X[:, i]
        correlation = np.corrcoef(xi, y)[0, 1]
        if correlation > 0.2:
            sign_vec[i] = 1
        elif correlation < -0.2:
            sign_vec[i] = -1
        else:
            sign_vec[i] = 0
    individuals = []
    c_masks = []
    # match_rates = []
    for _ in range(n):
        c_blocks = []

        for _ in range(c_num):

            x = np.zeros(n_features)
            for i in range(n_features):
                if sign_vec[i] == 1:
                    x[i] = np.random.uniform(low=0, high=upper_bound)
                elif sign_vec[i] == -1:
                    x[i] = np.random.uniform(low=lower_bound, high=0)
                else:
                    x[i] = np.random.uniform(low=lower_bound, high=upper_bound)
            
            # x += 0.1*np.random.randn(n_features)
            delta_x = x.reshape(-1, 1) - W_p
            c, *_ = np.linalg.lstsq(N, delta_x, rcond=None)

            c_blocks.append(c.flatten())

        individual = np.concatenate(c_blocks)
        individuals.append(individual)
        c_masks.append(np.random.randint(0, 2, c_num))
    
    individuals = np.array(individuals)  # shape: (n, c_num * r)
    c_masks = np.array(c_masks)  # shape: (n, c_num)

    return individuals, c_masks  # shape: (pop_size, c_num * r)

def initialize3(c_num, n, r, lower_bound=-5, upper_bound=5):
    
    individuals = np.random.uniform(low=lower_bound, high=upper_bound, size=(n, c_num * r))
    c_masks = np.zeros((n, c_num), dtype=int)

    mask_choices = ("all_ones", "single_one", "half_half")
    for i in range(n):
        mask_type = np.random.choice(mask_choices)
        if mask_type == "all_ones":
            mask = np.ones(c_num, dtype=int)
        elif mask_type == "single_one":
            mask = np.zeros(c_num, dtype=int)
            mask[np.random.randint(0, c_num)] = 1
        else:
            mask = np.zeros(c_num, dtype=int)
            num_ones = max(1, c_num // 2)
            ones_idx = np.random.choice(c_num, size=num_ones, replace=False)
            mask[ones_idx] = 1

        c_masks[i] = mask

    return individuals, c_masks


# 有什么好的进化算法吗？

# 这样的初始化方法可以保证 初始解就是一个帕累托图面了（假设性的）。  

# 当然也可以加上整数性的约束，让最后的无量纲数更加的简单易懂。



# 如果随机性的这种初始化方法在被选择的过程中没有保留，那么就说明这个方法就没什么作用。

