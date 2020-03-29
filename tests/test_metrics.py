import unittest
from utils.metrics import *


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


if __name__ == '__main__':
    unittest.main()
