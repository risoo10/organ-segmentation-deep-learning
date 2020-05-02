import h5py
import numpy as np
import unittest
from utils.dynamic_dataset import DynamicSet
from utils.utils import tensor_equals
import torch

def create_file():
    file = h5py.File('test.hdf5', 'w')
    file.create_dataset('test', data=[0])
    file.create_dataset('train', data=[1, 2])
    file.create_dataset('val', data=[2])
    for i in range(3):
        group = file.create_group(str(i))
        setx = group.create_dataset('x', dtype=np.float16, data=np.ones((100 * i + 1, 512, 512)) * i )
        sety = group.create_dataset('y', dtype=np.float16, data=np.ones((100 * i + 1, 512, 512)) * i)
        cropped = group.create_dataset('cropped', dtype=np.float16, data=[10, 90])
    return file

dfile = create_file()
single = torch.from_numpy(np.ones((1, 512, 512)).astype(np.float32))

class TestLitsSet(unittest.TestCase):

    def test_set_length(self):
        test_set = DynamicSet('test', dfile, cropped=False)
        train_set = DynamicSet('train', dfile, cropped=False)
        self.assertEqual(302, len(train_set))
        self.assertEqual(1, len(test_set))

    def test_set_get(self):
        train_set = DynamicSet('train', dfile, cropped=False)

        x, y, w = train_set[0]
        self.assertTrue(tensor_equals(x, single * 1))
        self.assertTrue(tensor_equals(y, single * 1))
        self.assertEqual(w.nelement(), 0)

        x, y, w = train_set[110]
        self.assertTrue(tensor_equals(x, single * 2))
        self.assertTrue(tensor_equals(y, single * 2))
        self.assertEqual(w.nelement(), 0)

    def test_set_get_weight(self):
        train_set = DynamicSet('train', dfile, cropped=False, weights=True)

        x, y, w = train_set[0]
        self.assertTrue(tensor_equals(x, single * 1))
        self.assertTrue(tensor_equals(y, single * 1))
        self.assertEqual(w.shape, (1, 512, 512))

    def test_set_get_sagital(self):
        train_set = DynamicSet('train', dfile, cropped=False, plane='sagital')

        x, y, w = train_set[0]
        self.assertEqual(x.shape, (1, 101, 512))
        self.assertEqual(y.shape, (1, 101, 512))

    def test_set_get_frontal(self):
        train_set = DynamicSet('train', dfile, cropped=False, plane='frontal')

        x, y, w = train_set[0]
        self.assertEqual(x.shape, (1, 101, 512))
        self.assertEqual(y.shape, (1, 101, 512))

    def test_set_get_cropped(self):
        train_set = DynamicSet('train', dfile, cropped=True)
        self.assertEqual(160, len(train_set))
