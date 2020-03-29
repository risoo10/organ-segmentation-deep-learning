import numpy as np


def dice_score(x, y):
    a = x.flatten()
    b = y.flatten()
    return (np.sum(a * b) * 2) / (np.sum(a) + np.sum(b))
