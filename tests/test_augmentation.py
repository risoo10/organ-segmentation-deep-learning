import unittest
from lits.lits_dataset import LitsDataSet
from lits.lits_set import LitsSet
from utils.constants import *
import numpy as np

dset = LitsDataSet("")
dset.slices = np.array([[10, 20], [40, 50], [90, 100]])
dset.cropped_slices = np.array([[11, 19], [41, 49], [91, 99]])
dset.x = np.ones((100, 512, 512))
dset.y = np.ones((100, 512, 512))


class TestSetAugmentation(unittest.TestCase):

    def test_augmentation_set(self):
        lits_set = LitsSet([1, 2], dset, augmentation=augmentation)
        x, y, w = lits_set[0]
        self.assertTrue(x.nelement() != 0)
        self.assertTrue(y.nelement() != 0)
