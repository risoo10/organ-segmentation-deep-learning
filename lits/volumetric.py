import torch
from torch.nn.functional import pad
import math
import numpy as np


def find_closest_power(x):
    a = math.ceil(math.log(x, 2))
    return 2**a


def find_max_shape(shapes):
    max_val = [0, 0, 0]
    for sh in shapes:
        for i in range(len(max_val)):
            if sh[i] > max_val[i]:
                max_val[i] = sh[i]

    return max_val


def variable_size_collate(batch):
    batch_size = len(batch)

    ac, aw, ah = find_max_shape([item[0].shape for item in batch])
    new_width = find_closest_power(aw)
    new_height = find_closest_power(ah)

    out_batch = [None] * batch_size

    for i in range(batch_size):

        # print(i, 'COLLATE', batch[i][0].shape, batch[i][1].shape, batch[i][2].shape)

        out_batch[i] = [None] * 3

        c, w, h = batch[i][0].shape

        pad_width = new_width - w
        pad_height = new_height - h
        padding = (0, pad_height, 0, pad_width)

        out_batch[i][0] = pad(batch[i][0], pad=padding, mode='constant', value=0)
        out_batch[i][1] = pad(batch[i][1], pad=padding, mode='constant', value=0)

        if batch[0][2].nelement() != 0:
            out_batch[i][2] = pad(batch[i][2], pad=padding,
                          mode='constant', value=0)
        else:
            out_batch[i][2] = batch[i][2]

        

        # print(i, 'AFTER PADDING', out_batch[i][0].shape, out_batch[i][1].shape, out_batch[i][2].shape)

    return torch.stack([item[0] for item in out_batch]), \
        torch.stack([item[1] for item in out_batch]), \
        torch.stack([item[2] for item in out_batch])
