import numpy as np


def accuracy(y_true, y_pred):
    hit = 0
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    for y in zip(y_pred, y_true):
        if y[0] == y[1]:
            hit += 1
    return (hit * 100) / y_pred.shape[0]
