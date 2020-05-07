from torch.utils.data import Dataset
from torchvision import models, transforms
import torch
from utils.utils import distance_transform_weight
import numpy as np
from utils.constants import *


class LitsSet(Dataset):
    def __init__(
            self,
            ind,
            dset,
            cropped=False,
            weights=False,
            transform=torch.from_numpy,
            classification=False,
            augmentation=None,
            plane="horizontal"
            sampler_weights=False
        ):
        self.ind = ind
        self.dset = dset
        self.cropped = cropped
        self.transform = transform
        self.weights = weights
        self.classification = classification
        self.augmentation = augmentation
        self.empty_weight = torch.Tensor()
        self.plane = plane

        if cropped:
            self.ind_map = self.crop_item_slice(ind)
        else:
            self.ind_map = self.map_item_slice(ind)

        self.length = len(self.ind_map)

        if self.sampler_weights:
            self.sampler_weights = self.load_sampler_weights()

    def __len__(self):
        return len(self.ind_map)

    def load_sampler_weights(self):
        w = self.dset.get_weights()
        if type(w) is np.ndarray:
            return w[self.ind_map]
        else:
            return None

    def __getitem__(self, i):
        ind = self.ind_map[i]
        _x = self.dset.x[ind].astype(np.float32)
        _y = self.dset.y[ind].astype(np.float32)

        if self.augmentation != None:
            aug = self.augmentation(image=_x, mask=_y)
            _x = aug["image"].astype(np.float32)
            _y = aug["mask"].astype(np.float32)

        x = _x[np.newaxis, :]

        if self.classification:
            y = np.array(np.any(_y)).astype(np.float32)
            return self.transform(x), torch.from_numpy(y), self.empty_weight
        else:
            y = _y[np.newaxis, :]

            if self.weights:
                weight = distance_transform_weight(_y).astype(np.float32)
                weight = np.nan_to_num(weight)
                weight = weight[np.newaxis, :].astype(np.float32)
                return self.transform(x), self.transform(y), self.transform(weight)
            else:
                return self.transform(x), self.transform(y), self.empty_weight

    def crop_item_slice(self, ind):
        ind_map = []
        for i in ind:
            pat_ind = self.dset.cropped_slices[i]
            self.plane_slice(ind_map, pat_ind)
        return ind_map

    def map_item_slice(self, ind):
        ind_map = []
        for i in ind:
            pat_ind = self.dset.slices[i]
            self.plane_slice(ind_map, pat_ind)
        return ind_map

    def plane_slice(self, ind_map, pat_ind):
        if self.plane == 'horizontal':
            self.horizontal_item_slice(ind_map, pat_ind)
        elif self.plane == 'frontal':
            self.vertical_item_slice(ind_map, pat_ind)
        elif self.plane == 'sagital':
            self.sagital_item_slice(ind_map, pat_ind)

    def horizontal_item_slice(self, ind_map, pat_ind):
        for x in range(pat_ind[0], pat_ind[1]):
            ind_map.append(x)

    def vertical_item_slice(self, ind_map, pat_ind):
        pat_slice = slice(pat_ind[0], pat_ind[1])
        all = slice(0,WIDTH)
        for i in range(WIDTH):
            ind_map.append((pat_slice, i, all))

    def sagital_item_slice(self, ind_map, pat_ind):
        pat_slice = slice(pat_ind[0], pat_ind[1])
        all = slice(0,HEIGHT)
        for i in range(HEIGHT):
            ind_map.append((pat_slice, all, i))
