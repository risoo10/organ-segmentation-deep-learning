import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score


def dice_score(x, y):
    a = x.flatten()
    b = y.flatten()
    return (np.sum(a * b) * 2) / (np.sum(a) + np.sum(b))


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self)
        self.name = 'Accuracy'

    def __call__(self, input, target):
        return accuracy_score(target, input)

class Recall(nn.Module):
    def __init__(self):
        super(Recall, self)
        self.name = 'Recall'

    def __call__(self, input, target):
        return recall_score(target, input, zero_division=1)