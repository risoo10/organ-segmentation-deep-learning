import numpy as np


def dice_score(X, Y, k):
    X = X.copy()
    Y = Y.copy()
    Y[Y == k] = 1
    X[X == k] = 1

    return np.sum(X[Y == 1]) * 2.0 / (np.sum(X) + np.sum(Y))
