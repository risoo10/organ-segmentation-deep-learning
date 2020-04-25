import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(true, inputs, eps=1e-7):
    true_f = true.contiguous().view(-1)
    inputs_f = inputs.contiguous().view(-1)
    intersection = (true_f * inputs_f).sum()
    union = true_f.sum() + inputs_f.sum()
    score = (2. * intersection + eps) / (union + eps)
    return (1 - score)


def tversky_loss(true, inputs, alpha, beta, eps=1e-7):
    y_true_pos = inputs.contiguous().view(-1)
    y_pred_pos = true.contiguous().view(-1)
    true_pos = (y_true_pos * y_pred_pos).sum()
    false_neg = (y_true_pos * (1-y_pred_pos)).sum()
    false_pos = ((1-y_true_pos) * y_pred_pos).sum()
    score = (true_pos + eps) / (true_pos + alpha*false_neg + beta*false_pos + eps)
    return (1 - score)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Dice'

    def forward(self, inputs, true):
        return dice_loss(true, inputs)


class TverskyLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.name = f'Tversy(a={beta})'
        self.alpha = 1 - beta
        self.beta = beta

    def forward(self, inputs, true):
        return tversky_loss(true, inputs, self.alpha, self.beta)


class WeightedLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.name = f'Weighted Dice {loss.name}'

    def forward(self, inputs, true, weights):
        iflat = inputs.contiguous().view(-1)
        wflat = weights.contiguous().view(-1)

        loss_part = self.loss(inputs, true)
        weight_part = torch.mean(iflat * wflat)

        return loss_part + weight_part
