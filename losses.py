import numpy as np


def categorical_cross_entropy(y_true, y_pred, derivative=False):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    y_true = np.clip(y_true, 1e-8, 1 - 1e-8)
    if not derivative:
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    else:
        return (y_pred - y_true) / y_pred.shape[0]


def mse(y_true, y_pred, derivative=False):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if not derivative:
        return np.mean(np.power(y_true - y_pred, 2))
    else:
        return 2 * (y_pred - y_true) / y_true.size
