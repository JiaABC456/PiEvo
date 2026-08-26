# PiEvo

PiEvo 是一个面向无量纲组合发现的进化搜索项目。它结合量纲分析、机器学习评估和多目标进化优化，从原始物理变量中自动搜索一组候选无量纲数，并在预测能力与组合独立性之间寻找平衡。

项目当前主要面向多类流体力学和传热相关数据集，例如 CHF、Rayleigh、Wall Model、Velocity Transformation、MHD、Keyhole 等。

## 项目目标

传统量纲分析通常依赖经验或手工构造无量纲数。这个项目希望通过程序自动完成以下事情：

- 根据量纲矩阵约束，生成满足量纲一致性的候选指数向量
- 将候选指数向量映射为无量纲组合
- 用机器学习模型评估这些组合对目标变量的预测能力
- 同时约束不同无量纲组合之间的冗余程度，鼓励得到更有区分度的组合
- 通过多目标进化算法得到一组 Pareto 最优候选解

## 方法概览

项目核心流程由 [`DEA.py`](/d:/PiEvo/DEA.py:1) 实现，可以概括为下面几个步骤：

1. 读取配置文件与数据集
2. 根据量纲矩阵分解出特解和零空间
3. 初始化候选种群
4. 将每个个体解码为若干个候选无量纲数
5. 计算两个优化目标
6. 通过非支配排序和拥挤距离进行选择
7. 迭代生成后代并更新种群
8. 输出最终结果、图像和解的分析文件

其中两个主要目标是：

- `MSE`：候选无量纲数组合对目标变量的预测误差
- `orth_loss`：候选无量纲数之间的相关性惩罚与其对目标变量相关性的折中指标

## 核心思想

### 1. 量纲约束

项目使用配置文件中的 `DIM_MATRIX` 表示量纲矩阵。对于目标变量和输入变量，程序先求一个特解，再通过零空间构造通解，从而保证生成的候选组合满足量纲一致性。

### 2. 候选无量纲数表示

每个个体包含两部分：

- 连续变量 `individual`：用于表示零空间中的系数
- 二值掩码 `c_mask`：用于表示哪些候选无量纲数被激活

这样做的好处是，模型既能搜索“指数怎么取”，也能搜索“保留几个无量纲数更合适”。

### 3. 多目标优化

项目不是只追求低预测误差，而是同时考虑：

- 候选组合能否有效预测目标
- 候选组合之间是否过于相似

因此最终输出不是单个答案，而是一组 Pareto 前沿解，方便后续人工筛选、物理解释或符号回归。

## 代码结构

项目主要文件如下：

- [`DEA.py`](/d:/PiEvo/DEA.py:1)：主算法实现，包含初始化、评估、进化、筛选、结果保存与可视化
- [`evaluate.py`](/d:/PiEvo/evaluate.py:1)：个体与种群评估逻辑
- [`data_Loss.py`](/d:/PiEvo/data_Loss.py:1)：回归误差、特征重要性和相关损失相关函数
- [`generate_data.py`](/d:/PiEvo/generate_data.py:1)：数据读取与部分数据生成逻辑
- [`init.py`](/d:/PiEvo/init.py:1)：种群初始化策略
- [`utils.py`](/d:/PiEvo/utils.py:1)：数据切分、非支配排序、拥挤距离、绘图等工具函数
- [`corr_show.py`](/d:/PiEvo/corr_show.py:1)：相关结果展示脚本
- [`comp_exp(s).py`](/d:/PiEvo/comp_exp(s).py:1)：对比实验脚本
- [`configs/`](/d:/PiEvo/configs:1)：各数据集对应的配置、原始数据和输出结果
- [`Baseline/`](/d:/PiEvo/Baseline:1)：用于对照或参考的外部方法与资料

## 数据集与配置

当前仓库中每个数据集通常对应一个独立目录，例如：

- `configs/CHF/`
- `configs/Rayleigh/`
- `configs/Velocity_transformation/`
- `configs/Wall_model/`
- `configs/MHD/`
- `configs/Keyhole/`

每个目录下通常包含：

- `config.json`：当前任务的参数配置
- 原始数据文件：如 `.csv`、`.txt`、`.mat`
- `pop_imgs/`：进化过程中的可视化图像
- `final_results/`：最终结果文件

一个典型 `config.json` 会包含以下参数：

- `POPULATION_SIZE`：种群规模
- `MAX_GENERATIONS`：最大进化代数
- `C_NUM`：候选无量纲数个数上限
- `DIM_MATRIX`：量纲矩阵
- `DATA_PATH`：数据文件路径
- `FUNCTION`：数据读取函数名
- `lower_bound`、`upper_bound`：搜索边界

## 运行环境

建议使用 Python 3.9 及以上版本。

项目核心依赖已经整理在 [`requirements.txt`](/d:/PiEvo/requirements.txt:1) 中。

## 快速开始

如果你只是想尽快把主流程跑起来，可以按下面步骤执行：

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 直接运行默认示例（Rayleigh 配置）：

```bash
python DEA.py
```

3. 如果想切换到其他数据集，修改 [`DEA.py`](/d:/PiEvo/DEA.py:696) 末尾的配置路径，例如：

```python
dea = DEA(config_file="./configs/CHF/config.json")
dea.run()
```

## 如何运行

目前主程序入口在 [`DEA.py`](/d:/PiEvo/DEA.py:696)，默认示例会运行 Rayleigh 配置。

直接运行：

```bash
python DEA.py
```

如果你想切换数据集，最简单的方法是修改 `DEA.py` 末尾的配置路径，例如：

```python
dea = DEA(config_file="./configs/CHF/config.json")
dea.run()
```

也可以在其他脚本中手动调用：

```python
from DEA import DEA

dea = DEA(config_file="./configs/Rayleigh/config.json")
dea.run()
```

## 输出结果说明

算法运行后，通常会在对应配置目录下生成以下内容：

### `pop_imgs/`

用于保存进化过程中的种群分布图，例如：

- `initial_population.png`
- `current_population.png`
- `final_population.png`
- `val_final_population.png`

这些图通常展示二维目标空间中的种群分布，并用颜色表示激活的无量纲数数量。

### `final_results/`

用于保存最终结果，例如：

- `results.npy`：每个个体的目标值
- `individuals.npy`：个体参数
- `c_masks.npy`：候选无量纲数组合掩码
- `solutions_analysis.csv`：候选解分析结果
- `pareto_front_solutions.csv`：Pareto 前沿解

## 当前实现特点

这个项目目前有几个比较明确的实现特点：

- 使用量纲矩阵零空间来保证候选解满足物理量纲约束
- 使用多种初始化策略，包括带符号先验的初始化
- 使用 LightGBM 评估候选无量纲数组合的预测能力
- 使用非支配排序与拥挤距离进行多目标筛选
- 支持多个数据集配置复用同一套搜索框架

## 适用场景

这个项目适合用于以下场景：

- 自动发现可能有物理意义的无量纲组合
- 为后续符号回归或机理建模提供特征候选
- 对比不同数据集上的无量纲搜索结果
- 研究预测性能与无量纲组合独立性之间的权衡关系

## 当前限制

从当前代码实现来看，也有一些需要注意的地方：

- 主入口目前写死在代码中，还没有统一的命令行参数接口
- 部分脚本与注释仍偏实验性质
- 数据读取函数中有一些路径是硬编码的
- README 中描述的是当前代码主干逻辑，个别实验脚本可能仍在演化中

## 后续可改进方向

如果后续继续维护，这个项目可以优先考虑下面几个方向：

- 增加命令行入口，支持直接选择配置文件
- 补充 `requirements.txt` 或 `environment.yml`
- 统一数据路径与输出目录管理
- 增加更完整的结果解释与可复现实验说明
- 将候选无量纲数的表达式导出得更直观，便于物理分析

## 致谢

仓库中的 [`Baseline/`](/d:/PiEvo/Baseline:1) 目录包含若干对照实现、参考工程或实验资料，可用于方法比较与思路借鉴。
