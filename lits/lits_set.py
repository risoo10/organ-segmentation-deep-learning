from torch.utils.data import Dataset
from torchvision import models, transforms
import torch

class LitsSet(Dataset):
    def __init__(self, ind, dset):
        self.ind = ind
        self.dset = dset
        self.ind_map = self.map_item_slice(ind)
        self.transform = torch.from_numpy

    def __len__(self): 
        return len(self.ind_map)

    def __get_item__(self, i):
        ind = self.ind_map[i]
        x = self.dset.x[ind]
        y = self.dset.y[ind]
        return self.transform(x), self.transform(y)

    def map_item_slice(self, ind):
        ind_map = []
        for i in ind:
            pat_ind = self.dset.slices[i]
            ind_map.extend(range(pat_ind[0], pat_ind[1]))

        return ind_map
