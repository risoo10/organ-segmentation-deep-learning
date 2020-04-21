import unittest
import torch
from lits.lits_set import *
from lits.lits_dataset import *
from utils.utils import tensor_equals

def create_dset():
    dset = LitsDataSet("")
    dset.slices = np.array([[10, 20], [40, 50], [90, 100]])
    dset.cropped_slices = np.array([[11, 19], [41, 49], [91, 99]])
    dset.x = np.ones((100, 512, 512))
    dset.y = np.ones((100, 512, 512))

    for i in range(100):
        dset.x[i, :, :] *= i
        dset.y[i, :, :] *= i

    single = torch.from_numpy(np.ones((1, 512, 512)).astype(np.float32))

    return dset, single

dset, single = create_dset()

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

    def test_cropped_slices(self):
        lits_set = LitsSet([1, 2], dset, cropped=True)
        self.assertEqual(lits_set.ind_map[0], 41)
        self.assertEqual(lits_set.ind_map[7], 48)
        self.assertEqual(lits_set.ind_map[8], 91)
        self.assertEqual(lits_set.ind_map[15], 98)

    def test_horizontal_slice(self):
        other_dset, other_single = create_dset()
        other_dset.x[:, 55, 55] = 0
        other_dset.y[:, 55, 55] = 2
        lits_set = LitsSet([1, 2], other_dset, plane="horizontal")
        
        ind = lits_set.ind_map[9]
        x = other_dset.x[ind]
        y = other_dset.y[ind]
        print('Horizontal slice', x.shape, y.shape, ind)
        self.assertEqual(x[55, 55], 0)
        self.assertEqual(y[55, 55], 2)

    def test_vertical_slice(self):
        other_dset, other_single = create_dset()
        other_dset.x[:, :, 55] = 0
        other_dset.y[:, :, 55] = 2
        lits_set = LitsSet([1, 2], other_dset, plane="vertical")
        
        ind = lits_set.ind_map[9]
        x = other_dset.x[ind]
        y = other_dset.y[ind]
        print('Vertical slice', x.shape, y.shape, ind)
        self.assertEqual(x[0, 55], 0)
        self.assertEqual(y[0, 55], 2)

    def test_sagital_slice(self):
        other_dset, other_single = create_dset()
        other_dset.x[:, 55, :] = 0
        other_dset.y[:, 55, :] = 2
        lits_set = LitsSet([1, 2], other_dset, plane="sagital")
        
        ind = lits_set.ind_map[9]
        x = other_dset.x[ind]
        y = other_dset.y[ind]
        print('Sagital slice', x.shape, y.shape, ind)
        self.assertEqual(x[0, 55], 0)
        self.assertEqual(y[0, 55], 2)

