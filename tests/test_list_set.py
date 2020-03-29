import unittest
from lits.lits_set import *
from lits.lits_dataset import *


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
        self.assertEqual(lits_set.__get_item__(0), (400, 4000))
        self.assertEqual(lits_set.__get_item__(9), (490, 4900))
        self.assertEqual(lits_set.__get_item__(10), (900, 9000))
        self.assertEqual(lits_set.__get_item__(19), (990, 9900))

    def test_set_mapper(self):
        lits_set = LitsSet([1, 2], dset)
        self.assertEqual(lits_set.ind_map[0], 40)
        self.assertEqual(lits_set.ind_map[9], 49)
        self.assertEqual(lits_set.ind_map[10], 90)
        self.assertEqual(lits_set.ind_map[19], 99)


