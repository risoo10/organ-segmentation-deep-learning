import unittest
import torch
from lits.lits_set import *
from lits.lits_dataset import *
from utils.utils import tensor_equals


dset = LitsDataSet("")
dset.slices = np.array([[10, 20], [40, 50], [90, 100]])
dset.x = np.arange(0, 100).reshape((-1, 1)).astype(np.float32) * 10
dset.y = np.arange(0, 100).reshape((-1, 1)).astype(np.int32) * 100


class TestLitsSet(unittest.TestCase):

    def test_set_length(self):
        lits_set = LitsSet([1, 2], dset)
        self.assertEqual(lits_set.__len__(), 20)

    def test_set_get(self):
        lits_set = LitsSet([1, 2], dset)
        self.assertEqual(lits_set.__get_item__(0), (400, 4000, None))
        self.assertEqual(lits_set.__get_item__(9), (490, 4900, None))
        self.assertEqual(lits_set.__get_item__(10), (900, 9000, None))
        self.assertEqual(lits_set.__get_item__(19), (990, 9900, None))

    def test_set_get_weights(self):
        dset = LitsDataSet("")
        dset.slices = np.array([[0, 10]])
        dset.x = np.zeros((10, 10, 10)).astype(np.float32)
        dset.y = np.zeros((10, 10, 10)).astype(np.int32)
        # dset.x[:, 2:6, 2:6] = 1
        # dset.y[:, 2:6, 2:6] = 1
        lits_set = LitsSet([0], dset, weights=True)
        
        x,y,w = lits_set.__get_item__(0)

        print(x.dtype, y.dtype, w.dtype)

        _x = torch.from_numpy(np.zeros((10, 10), np.float32))
        _y = torch.from_numpy(np.zeros((10, 10), np.int32))
        self.assertTrue(tensor_equals(x, _x))
        self.assertTrue(tensor_equals(y, _y))
        self.assertEqual(w.shape, (10, 10))

    def test_set_mapper(self):
        lits_set = LitsSet([1, 2], dset)
        self.assertEqual(lits_set.ind_map[0], 40)
        self.assertEqual(lits_set.ind_map[9], 49)
        self.assertEqual(lits_set.ind_map[10], 90)
        self.assertEqual(lits_set.ind_map[19], 99)

