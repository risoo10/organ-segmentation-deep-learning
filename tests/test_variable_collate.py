import unittest
from lits.volumetric import variable_size_collate
from torch.utils.data import Dataset, DataLoader
from lits.lits_dataset import LitsDataSet
from lits.lits_set import LitsSet
from random import randint
import numpy as np

dset = LitsDataSet("")
dset.slices = np.array([[10, 20], [40, 50], [90, 100]])
dset.cropped_slices = np.array([[11, 19], [41, 49], [91, 99]])

random_list = [np.ones((randint(257, 512), randint(257, 512))) for i in range(100)]
dset.x = np.array(random_list, dtype=object)
dset.y = np.array(random_list, dtype=object)

class TestVolumetricCollate(unittest.TestCase):

    def test_variable_size_collate(self):
        lits_set = LitsSet([1, 2], dset)
        dataloader = DataLoader(lits_set, batch_size=2, shuffle=True, num_workers=0, collate_fn=variable_size_collate)

        test_iterator = iter(dataloader)
        a = test_iterator.next()
        x, y, w = a
        print('Finished Collate', x.shape, y.shape, w.shape) 
        self.assertEqual(x.shape[0], 2)
        self.assertEqual(x.shape[1], 1)
        self.assertEqual(x.shape[2], 512)
        self.assertEqual(x.shape[3], 512)


