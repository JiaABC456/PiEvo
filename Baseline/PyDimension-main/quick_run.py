

import sys
from pathlib import Path
import argparse
import numpy as np
# Add pydimension to path
sys.path.insert(0, str(Path(__file__).parent))

from pydimension.dimensional_analysis import DimensionalAnalyzer, DimensionalAnalysisConfig
from pydimension.optimization_discovery import OptimizationDiscoverer, OptimizationDiscoveryConfig
import pandas as pd


# parser = argparse.ArgumentParser(
#     description='Run the complete PyDimension 2.0 pipeline',
#     formatter_class=argparse.RawDescriptionHelpFormatter,
#     epilog='Example usage: python quick_run.py --config config.json'
# )


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

X = np.random.rand(100,5)
y = np.random.rand(100)
variable_names = [f"var_{i}" for i in range(X.shape[1])] + ["target"]
normalized_data = normalize_data(X,y)
full_matrix = np.array([
    [1, -2, 1, 2, 2, 0],
    [0, 1, 1, -1, 3, 0],
    [-2, 3, 1, 1, 2, 0],
    [0, 0, 1, 0, 0, 0],
])
# 我只需要这样就好了
config = DimensionalAnalysisConfig()
analyzer = DimensionalAnalyzer(config)
afterdata, basis_vectors = analyzer.process(normalized_data, full_matrix, variable_names)
data = analyzer.compute_normalized_lg_pis()



config = OptimizationDiscoveryConfig()
optimizer = OptimizationDiscoverer(config)
optimizer.process(data, basis_vectors)
result = optimizer._construct_discovered_equation()
print(result)

