import numpy as np


def dice_score(x, y, k):
    x = x.copy()
    y = y.copy()
    y[y == k] = 1
    x[x == k] = 1

    return np.sum(x[y == 1]) * 2.0 / (np.sum(x) + np.sum(y))
