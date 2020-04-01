from torch.utils.data import Dataset
from torchvision import models, transforms
import torch
from utils.utils import distance_transform_weight
import numpy as np

class LitsSet(Dataset):
    def __init__(self, ind, dset, weights=False):
        self.ind = ind
        self.dset = dset
        self.ind_map = self.map_item_slice(ind)
        self.transform = torch.from_numpy
        self.weights = weights

    def __len__(self):
        return len(self.ind_map)

    def __get_item__(self, i):
        ind = self.ind_map[i]
        x = self.dset.x[ind]
        y = self.dset.y[ind]

        if self.weights:
            weight = distance_transform_weight(y).astype(np.float32)
            weight = np.nan_to_num(weight)
            return self.transform(x), self.transform(y), self.transform(weight)
        else:
            return self.transform(x), self.transform(y), None

    def map_item_slice(self, ind):
        ind_map = []
        for i in ind:
            pat_ind = self.dset.slices[i]
            ind_map.extend(range(pat_ind[0], pat_ind[1]))

        return ind_map
