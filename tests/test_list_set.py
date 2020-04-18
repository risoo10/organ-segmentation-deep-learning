import unittest
import torch
from lits.lits_set import *
from lits.lits_dataset import *
from utils.utils import tensor_equals


dset = LitsDataSet("")
dset.slices = np.array([[10, 20], [40, 50], [90, 100]])
dset.x = np.ones((100, 512, 512))
dset.y = np.ones((100, 512, 512))

for i in range(100):
    dset.x[i, :, :] *= i
    dset.y[i, :, :] *= i

single = torch.from_numpy(np.ones((1, 512, 512)).astype(np.float32))


class TestLitsSet(unittest.TestCase):

    def test_set_length(self):
        lits_set = LitsSet([1, 2], dset)
        self.assertEqual(lits_set.__len__(), 20)

    def test_set_get(self):
        lits_set = LitsSet([1, 2], dset)

        x, y, w = lits_set.__getitem__(0)
        self.assertTrue(tensor_equals(x, single * 40))
        self.assertTrue(tensor_equals(y, single * 40))
        self.assertEqual(w.nelement(), 0)

        x, y, w = lits_set.__getitem__(19)
        self.assertTrue(tensor_equals(x, single * 99))
        self.assertTrue(tensor_equals(y, single * 99))
        self.assertEqual(w.nelement(), 0)
        

    def test_set_get_weights(self):
        lits_set = LitsSet([0], dset, weights=True)
        x, y, w = lits_set.__getitem__(0)

        print(x.dtype, y.dtype, w.dtype)

        self.assertTrue(tensor_equals(x, single * 10))
        self.assertTrue(tensor_equals(y, single * 10))
        self.assertEqual(w.shape, (1, 512, 512))

    def test_set_mapper(self):
        lits_set = LitsSet([1, 2], dset)
        self.assertEqual(lits_set.ind_map[0], 40)
        self.assertEqual(lits_set.ind_map[9], 49)
        self.assertEqual(lits_set.ind_map[10], 90)
        self.assertEqual(lits_set.ind_map[19], 99)
