from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(actual_array, model_array):
    return np.sqrt(mean_squared_error(actual_array, model_array))