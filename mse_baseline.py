

from generate_data import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from data_Loss import *

X, y, _ = handle_Wall_model("_")

X_train, y_train = X[:int(X.shape[0]*0.8),:], y[:int(y.shape[0]*0.8)]

X_val, y_val = X[int(X.shape[0]*0.8):,:], y[int(y.shape[0]*0.8):]


mse, r2, _ = lgb_mse(X_train, y_train, X_val, y_val)

print(f"Validation MSE: {mse:.6f}, R2: {r2:.6f}")