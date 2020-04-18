from torch.utils.data import Dataset
from torchvision import models, transforms
import torch
from utils.utils import distance_transform_weight
import numpy as np
from utils.constants import *

class LitsSet(Dataset):
    def __init__(self, ind, dset, cropped=False, weights=False, transform=torch.from_numpy):
        self.ind = ind
        self.dset = dset
        self.cropped = cropped
        self.transform = transform
        self.weights = weights
        
        if cropped:
            self.ind_map = self.crop_item_slice(ind)
        else:
            self.ind_map = self.map_item_slice(ind)

        self.length = len(self.ind_map)

    def __len__(self):
        return len(self.ind_map)

    def __getitem__(self, i):
        ind = self.ind_map[i]
        x = self.dset.x[ind].reshape((1, WIDTH, HEIGHT)).astype(np.float32)
        _y = self.dset.y[ind]
        y = _y.reshape((1, WIDTH, HEIGHT)).astype(np.float32)

        if self.weights:
            weight = distance_transform_weight(_y).astype(np.float32)
            weight = np.nan_to_num(weight).reshape((1, WIDTH, HEIGHT)).astype(np.float32)
            return self.transform(x), self.transform(y), self.transform(weight)
        else:
            return self.transform(x), self.transform(y), None

    def crop_item_slice(self, ind):
        ind_map = []
        for i in ind:
            pat_ind = self.dset.cropped_slices[i]
            for x in range(pat_ind[0], pat_ind[1]):
                ind_map.append(x)

        return ind_map


    def map_item_slice(self, ind):
        ind_map = []
        for i in ind:
            pat_ind = self.dset.slices[i]
            for x in range(pat_ind[0], pat_ind[1]):
                ind_map.append(x) 
        return ind_map
