import unittest
import torch
from torch.nn import BCELoss
from utils.losses import DiceLoss, TverskyLoss, WeightedLoss
from utils.utils import tensor_equals
import numpy as np


class TestMetrics(unittest.TestCase):
    # BCE LOSS
    def test_bce(self):
        loss = BCELoss()
        x = torch.FloatTensor([[0., 1., 1., 0.]])
        y = torch.FloatTensor([[0., 1., 1., 0.]])
        score = loss(x, y)
        print('BCE', score)
        self.assertTrue(torch.eq(score, 0))

    def test_bce_flat(self):
        loss = BCELoss()
        x = torch.FloatTensor([[0., 1.], [1., 0.]])
        y = torch.FloatTensor([[0., 1.], [1., 0.]])
        score = loss(x, y)
        print('BCE', score)
        self.assertTrue(torch.eq(score, 0))


    # DICE LOSS
    def test_dice(self):
        dice_loss = DiceLoss()
        x = torch.FloatTensor([[0., 1., 1., 0.]])
        y = torch.FloatTensor([[0., 1., 1., 0.]])
        dice = dice_loss(x, y)
        print('DICE', dice)
        self.assertTrue(torch.eq(dice, 0))

    def test_dice_flat(self):
        dice_loss = DiceLoss()
        x = torch.FloatTensor([[0., 1.], [1., 0.]])
        y = torch.FloatTensor([[0., 1.], [1., 0.]])
        dice = dice_loss(x, y)
        print('DICE', dice)
        self.assertTrue(torch.eq(dice, 0))

    def test_dice_negatives(self):
        dice_loss = DiceLoss()
        x = torch.FloatTensor([[1., 1.], [1., 0.]])
        y = torch.FloatTensor([[0., 1.], [1., 0.]])
        dice = dice_loss(x, y)
        self.assertEqual(round(dice.item(), 2), 0.2)


    # TVERSKY LOSS
    def test_tversky(self):
        loss = TverskyLoss(beta=0.5)
        x = torch.FloatTensor([[1., 1.], [1., 0.]])
        y = torch.FloatTensor([[0., 1.], [1., 0.]])
        score = loss(x, y)
        self.assertEqual(round(score.item(), 2), 0.2)

    def test_tversky_false_posites(self):
        loss = TverskyLoss(beta=0.3)
        x = torch.FloatTensor([[1., 1.], [1., 0.]])
        y = torch.FloatTensor([[0., 1.], [1., 0.]])
        score = loss(x, y)
        self.assertEqual(round(score.item(), 2), 0.26)

    def test_tversky_false_negatives(self):
        loss = TverskyLoss(beta=0.7)
        x = torch.FloatTensor([[0., 0.], [1., 1.]])
        y = torch.FloatTensor([[0., 1.], [1., 1.]])
        score = loss(x, y)
        self.assertEqual(round(score.item(), 2), 0.26)


    # WEIGHTED LOSSES
    def test_weighted_dice(self):
        loss = DiceLoss()
        weighted_loss = WeightedLoss(loss)
        x = torch.FloatTensor([[0., 1.], [1., 1.]])
        y = torch.FloatTensor([[0., 0.], [1., 1.]])
        w = torch.FloatTensor([[0.25, 0.25], [0., 0.]])
        score = weighted_loss(x, y, w)
        self.assertEqual(round(score.item(), 2), 0.26)


    def test_weighted_tversky(self):
        loss = TverskyLoss(beta=0.3)
        weighted_loss = WeightedLoss(loss)
        x = torch.FloatTensor([[0., 1.], [1., 1.]])
        y = torch.FloatTensor([[0., 0.], [1., 1.]])
        w = torch.FloatTensor([[0.25, 0.25], [0., 0.]])
        score = weighted_loss(x, y, w)
        self.assertEqual(round(score.item(), 2), 0.32)


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
