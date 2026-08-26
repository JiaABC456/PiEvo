import pandas as pd
import numpy as np
from scipy.linalg import null_space
from data_Loss import sparse_poly_mse, rf_mse
from utils import sklearn_split_method_with_indices, remove_bad_rows

df = pd.read_csv('chf_public.csv')
# 删除前两列和最后一列
df = df.iloc[:, 2:-1]

# 在倒数第二列插入重力加速度列（常数）
g_value = 9.81
df.insert(len(df.columns) - 1, 'g', g_value)

# 自变量 X
X = df.iloc[1:1000, :-1].values.astype(float)  # 所有列除了最后一列
# 因变量 y
y = df.iloc[1:1000, -1].values.astype(float)  # 最后一列

X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = sklearn_split_method_with_indices(X, y)

dim_matrix = np.array([
    # D   L   P   G   x   h_sub  T_in  g   q_chf
    [ 0,  0,  1,  1,  0,    0,     0,   0,    1 ],   # M
    [ 1,  1, -1, -2,  0,    2,     0,   1,    0 ],   # L
    [ 0,  0, -2, -1,  0,   -2,     0,  -2,   -3 ],   # T
    [ 0,  0,  0,  0,  0,    0,     1,   0,    0 ]    # Theta
])

c_num = 4  # 10个无量纲数

D = dim_matrix[:, :-1]      # k x n
q = dim_matrix[:, -1].reshape(-1, 1)      # k x 1

W_p, residuals, rank, s = np.linalg.lstsq(D, q, rcond=None)  # n x 1
print(rank)
W_p_expanded_repeat = np.repeat(W_p, c_num, axis=1)
N = null_space(D)   # shape: (n, r)
r = N.shape[1]
# c = np.array([0 for i in range(r)]).reshape(-1, 1)  # r x 1
# 通解
# x = W_p + N @ c   # sp.Matrix(c) 变成列向量 n*1   

# 要优化的变量
c_num_vector = np.array([1 for i in range(c_num)]).reshape(-1, 1)
c_vector = np.array([1 for i in range(c_num*r)]).reshape(-1, 1)

# 得到最后的x
repeated_c_vector = np.repeat(c_num_vector, r, axis=0)
masked_c_vector = c_vector * repeated_c_vector # c_num * r
x = W_p_expanded_repeat + N @ masked_c_vector.reshape(r, -1)   # n* c_num

# print(x.shape) # n x c_num
# preds = np.prod(X ** x[:,1].T, axis=1)  # c_num * n
# print(preds.shape)  # c_num * m
preds = []
for i in range(c_num):
    xi = x[:, i].reshape(-1, 1)  # n x 1
    pred_i = np.prod(X ** xi.T, axis=1)  # m x 1
    preds.append(pred_i)
preds = np.column_stack(preds)  # m x c_num

# preds_train, y_train = remove_bad_rows(preds[idx_train], y[idx_train])
# preds_test, y_test = remove_bad_rows(preds[idx_test], y[idx_test])


# mse,r2 = rf_mse(preds_train, y_train, preds_test, y_test)
mse, r2, _ = rf_mse(X_train, y_train, X_test, y_test)
print("Random Forest MSE:", mse)
print("Random Forest R2:", r2)
# print(rf.feature_importances_)

# m x c_num
# 一个是实数优化 一个是0-1优化。
# 为啥不能是整数呢?  最终就是很少的无量纲数

# 怎么初始化
# 怎么选择解
# 怎么更新解
# 最后怎么生成最后的解  比如我10个最后降到两个。
# 可以参考SparseEA，在这个基础上加上表格数据没有的。  加速收敛或者多样性




# 初始化方法 ： 找到一个c，拿进化算法优化，让mse最小化，得到的c作为初始解之一。
# 其他的c就在周围进行初始化？
# 定量还有定方向
# c_mask 0-1只有对应的c为1。其他都为0。

# 定方向
# 初始化方法2： 根据特征与y的相关性来判断正和负相关，从而保证c的选择，来确定指数的正负号。
# c_mask 0-1百分之50.

# 什么也不定，完全随机初始化。
# 初始化方法3： 完全随机初始化。 
# c_mask都是1

# c_mask的初始化的时候要考虑每个无量纲数的重要性。无量纲数越大，c_mask越为1.


# 交叉算子： pi之间去相关性的交叉算子。




# 写 data loss 写一个损失函数
# 一个过程 把模型的选择也加载这个选择中
# 交叉损失等等
def data_loss(c, N, W_p, data):
    x = W_p + N @ c
    # 计算预测值
    preds = np.prod(data ** x.T, axis=1)
    # 计算真实值
    true_vals = data['CHF'].values
    # 计算均方误差损失
    loss = np.mean((preds - true_vals) ** 2)
    return loss

# 这个数比如说是10个无量纲数，也就是10*r个向量。 再加一个选择什么样的模型来拟合？

# c的个数是要进行稀疏优化的。  如果不对 就直接把对应的数变为0。

# 初始化操作怎么做？

# 把包含因变量的所有无量纲数得到行不行？ 也就是要执行多少次优化？

# c的数量作为一个优化目标 我就取最简单的多项式对应的公式 数据损失。

# 因为你c不一样，说明你选择的无量纲数不一样。  你选择的无量纲数不一样，说明你选择的模型不一样。


# 初始化操作： 比如我初始化有10个c，也就是50个变量。
# 挑选出10个最能保证数据损失最小的c。其他的都随机。来初始化种群。

# 在多项式优化的过程中得到最优的参数，可以指导c稀疏性的选择。
# 比如我最后得到了一个seita。我可以通过这个seita来评估每个无量纲数的重要性。
# 重要性低的无量纲数对应的c可以考虑置零。然后把这个添加到优化池中。

# 最后还是 最主要的是优化c和c的稀疏性，这个过程中还包含如何优化多项式参数。



# 还是在稀疏进化算法的基础上来求解。   这个过程的目标函数是什么呢？只有一个数据损失吗？

# 最后你肯定是为了指导c。
# 这里面肯定也要有c的初始化，mask的初始化过程。



# 这个过程还真要把稀疏度作为一个目标函数了。





# 评价一个解的好坏：1. mse 这个是最主要的 2. 正交性



# 关于使用随机森林还是使用laosso回归的问题。
# 随机森林更适合处理非线性关系，而且可以自动处理特征选择问题。Lasso回归则更适合线性关系，并且通过L1正则化实现特征选择。
# 如果数据中存在复杂的非线性关系，随机森林可能表现更好；  大概就是前期使用随机森林来评估特征重要性，后期使用Lasso回归来进行最终的模型简化和解释。

# 在后面建立loss和c之间的关系，用贝叶斯选点的方式来优化c。  应该是c*mask这个值和mse之间的关系。
# 建立这个关系需要快速，而且不能每次都建立，建立的时候怎么选择点也是一个问题。
# 这个过程也可以得到对现在c的更新。
# 这个就变成了两个情况，第一种是用贝叶斯选点的方式来选择新的c，第二种情况是模型提供一个方向来更新现在的c。



# 关于更新c_mask的问题。
# 一个无量纲数重不重要，要看它的特征重要性，重要就是1，不重要就是0，这是显然的。
# 而且看这个无量纲数和其他无量纲数之间的相关性。相关性高就设置为0，否则设置为1。
# 对一整个c，评价使用正交性损失函数进行评价。
# P(c_mask = 1) = 重要性*相关性。
# 这个过程好像本身就能导致c_mask稀疏。

# 关于c_mask的稀疏性问题。也就是保证c_mask最小，最完备的无量纲组。

# 关于初始化问题。那就是得到正交性和数据损失都包含了。
# 第一个就是把每个种群中最好的解都提取出来，然后10个10个的装在一起，形成一个c。 他们的正交性肯定很高。
# 随机设置c_mask为1。


# 当重要性都相同的时候，就说明大家都一样，到不了选谁的问题，就随机选。

# 希望这个进化的过程就是把信息从多个无量纲数上往少数的，独立的无量纲数上聚合的过程。


# 如果几万条的数据不行的话，
# 那么我就取一点数据，然后这些数据拟合到一个神经网络中。


# 初始化时候的每个初始化的种群大小是可以进行优化的。

# c_num不能太大，太大可能会导致升维过程


# 我得到每个的相关性和重要性，根据相关性和重要性的乘积来决定c_mask的取值概率。
# 相关性低和重要性高，说明这个无量纲数很重要，保留吗？ 
# 相关性高和重要性低，说明这个无量纲数不重要，去除吗？
# 那这个过程就是只对1的下手，而且1变为0，0会越来越多。

# 那有一个问题，就是当都变为0的时候怎么办？只有一个1的时候怎么办？

# 两个必须放在同一个尺度下考虑。
# 这个过程必须25%的概率由0变为1，25%的概率由1变为0。

# 在变异的时候考虑稀疏性和正交性。 这个是对c_mask的变异。
# 对c的变异就是方向性的变异。

# 在选择的时候当然使用非支配排序选择了。
# 我怎么感觉目标函数唯一起作用的地方是非支配排序这个地方了。其他地方起作用吗？
# 什么时候会淘汰解，不就是非支配排序的时候吗？
# 选择是怎么选择的？

# 首先是父类 得到父类之后，进行选择得到父类。
# 然后进行交叉和变异，得到子类。
# 然后把父类和子类放在一个池子里面进行非支配排序选择，选择出下一代的种群。


# 你可以出现全1的，也就是复杂，而且mse小的。
# 你也可以做一个只有1的，也就是简单，但是mse大的。
# 你也可以是各个0.5的，这个才是随机生成。

# 第二个就是符号的

# 第三个GA专门优化一个无量纲数的。把这个每个解嵌入到前两个初始化种群中去。
# 所以就变成了三种初始化方法的结合体。

# loss越大说明，

# data_loss越小 正交性损失越大 data_loss越大 正交性损失越小。


# 解决进化算法慢的问题，解决数据条数少的问题。 
# 主要还是rf运行慢导致的。 
# 最后还有一个pysr模型收尾。






# 如何解决过拟合问题呢？ 改变一下mse，把它让他接近rf的mse。可以低，越低惩罚越大。是一个渐进的过程。

# 最重要的是，进行过拟合的分析，什么时候mse小就达到了过拟合的程度，这个mse就可以融入到mse中。

# 所以说现在的问题是：1. 大数据集快速评估问题 2. 过拟合问题 3. c_mask的稀疏性问题
# 把这几个解决了，这篇论文就差不多了。
# 变异算子这个最有搞头。


# 不起作用的原因就是 c的变化导致Pi的变化，因此导致每次feature_importances都不一样。
# 这样就没法指导c_mask的变化了。
# 用c来预测最后得到的特征重要性。
# 或者用c来预测c_mask。
# 这个用c来预测特征重要性好像也没啥区别吧。

# 假如想象一下，把所有的c都取到，得到的是无数个无量纲数。
# 这个就变成了每一列都不变了。
# 这样的话，每一列的finess就不会变了。
# 所以这个就退化成了传统的背包问题。
# 但是我不可能穷尽所有的c。


# 它也是有一个c的，就是让这个c和一个固定的数据集进行交互，评价的每个c的重要性。
# 我现在评价不了，我的问题不是让c稀疏，而是让c产生的pi稀疏。
# 所以，现在的任务就是让c预测产生各个pi的重要性。
# 这样就能指导c_mask的变化了。
# 什么情况下能让c_num提高呢？什么样的数据集可以呢？

# 一个问题SparseEA在当dec和mask分开的时候是不起作用的。


# 现在的问题就变成了：我用c预测特征重要性。而且我希望这个特征重要性不要变。
# 其实这个特征重要性可以用，一个c产生的无量纲数与y的相关性表示。

# 是不是应该先有一个相互冲突的目标函数，再进行变异交叉算子的研发？
# 在一个系统中得到的特征重要性绝对不能当作一个信息来指导c和c_mask的变化。



# 需要再改一改损失函数 让每个为1的无量纲之间的不能被其他无量纲数表示。


# 两个问题： 第一个： 改损失函数 第二个：整数的约束 这个整数的约束不知道能不能作为另一个目标函数？

# 随机子集？

#让“贡献小的维度”去承担多样性成本