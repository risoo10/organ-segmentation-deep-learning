import unittest
from lits.lits_dataset import *


class TestLitsDataset(unittest.TestCase):

    def test_dataset_split(self):
        dset = LitsDataSet("")
        train, test, val = dset.split_test_val_train(10)
        self.assertEqual(len(train), 7)
        self.assertEqual(len(test), 2)
        self.assertEqual(len(val), 1)

