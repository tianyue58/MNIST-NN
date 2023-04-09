import numpy as np


def relu(x, derivative=False):
    if not derivative:
        return x * (x > 0)
    else:
        return 1 * (x > 0)


def sigmoid(x, derivative=False):
    if not derivative:
        return 1 / (1 + np.exp(-x))
    else:
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)


def tanh(x, derivative=False):
    if not derivative:
        return np.tanh(x)
    else:
        return 1 - np.tanh(x) ** 2


def softmax(x, derivative=False):
    if not derivative:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        exp_x = np.exp(x)
        sm = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return sm * (1 - sm)
