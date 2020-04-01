import unittest
import torch
from utils.metrics import *
from utils.utils import tensor_equals


class TestMetrics(unittest.TestCase):
    def test_dice(self):
        x = np.array([0., 1., 1., 0.])
        y = np.array([0., 1., 1., 0.])
        dice = dice_score(x, y)
        self.assertEqual(dice, 1)

    def test_dice_flat(self):
        x = np.array([[0., 1.], [1., 0.]])
        y = np.array([[0., 1.], [1., 0.]])
        dice = dice_score(x, y)
        self.assertEqual(dice, 1)

    def test_tensor_equals(self):
        a = torch.from_numpy(np.ones((10, 10)))
        b = torch.from_numpy(np.ones((10, 10)))
        self.assertTrue(tensor_equals(a, b))

    def test_tensor_equals_falsy(self):
        a = torch.from_numpy(np.ones((10, 10)))
        b = torch.from_numpy(np.zeros((10, 10)))
        self.assertFalse(tensor_equals(a, b))

if __name__ == '__main__':
    unittest.main()
