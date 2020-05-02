import h5py
from utils.constants import *
import numpy as np
import pickle
from torch.utils.data import Dataset
from utils.utils import distance_transform_weight


class Slice:
    def __init__(self, id, array_slice):
        self.id = id
        self.array_slice = array_slice


class DynamicSet(Dataset):
    def __init__(
        self,
        set,
        dset_file,
        cropped=False,
        weights=False,
        transform=torch.from_numpy,
        classification=False,
        augmentation=None,
        plane="horizontal"
    ):

        assert set in ['train', 'test', 'val']
        self.set = set
        ind = dset_file[set][:].astype(np.int)
        self.dset_file = dset_file
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

    def __len__(self):
        return len(self.ind_map)

    def __getitem__(self, i):
        ind = self.ind_map[i]
        x = self.dset_file[ind.id]['x'][ind.array_slice]
        x = x[np.newaxis, :].astype(np.float32)
        _y = self.dset_file[ind.id]['y'][ind.array_slice]

        if self.augmentation != None:
            aug = self.augmentation(image=x, mask=_y)
            x = aug["image"]
            _y = aug["mask"]

        if self.classification:
            y = np.array(np.any(_y)).astype(np.float32)
            return self.transform(x), torch.from_numpy(y), self.empty_weight
        else:
            y = _y[np.newaxis, :].astype(np.float32)

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
            id = str(i)
            boundaries = self.dset_file[id]['cropped'][:].astype(np.int)
            self.plane_slice(id, boundaries, ind_map)
        return ind_map

    def map_item_slice(self, ind):
        ind_map = []
        for i in ind:
            id = str(i)
            total = self.dset_file[id]['x'].shape[0]
            boundaries = [0, total]
            self.plane_slice(id, boundaries, ind_map)
        return ind_map

    def plane_slice(self, id, boundaries, ind_map):
        if self.plane == 'horizontal':
            self.horizontal_item_slice(ind_map, id, boundaries)
        elif self.plane == 'frontal':
            self.vertical_item_slice(ind_map, id, boundaries)
        elif self.plane == 'sagital':
            self.sagital_item_slice(ind_map, id, boundaries)

    def horizontal_item_slice(self, ind_map, id, boundaries):
        all = slice(0, WIDTH)
        for x in range(boundaries[0], boundaries[1]):
            ind_map.append(
                Slice(id, (x, all, all))
            )

    def vertical_item_slice(self, ind_map,  id, boundaries):
        all = slice(0, WIDTH)
        z_slice = slice(boundaries[0], boundaries[1])
        for i in range(WIDTH):
            ind_map.append(Slice(id, (z_slice, i, all)))

    def sagital_item_slice(self, ind_map,  id, boundaries):
        z_slice = slice(boundaries[0], boundaries[1])
        all = slice(0, HEIGHT)
        for i in range(HEIGHT):
            ind_map.append(Slice(id, (z_slice, all, i)))
