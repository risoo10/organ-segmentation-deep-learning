import unittest
import torch
import numpy as np
from inception.transforms import CatTransform, CustomResize, ReshapeTransform
from utils.constants import *
from torchvision.transforms import Compose


class TestLitsSet(unittest.TestCase):
    def test_incpetion_transforms(self):
        resize_transform = Compose([
            torch.from_numpy,
            ReshapeTransform((1, 1, WIDTH, HEIGHT)),
            CustomResize(size=(299, 299)),
            ReshapeTransform((1, 299, 299)),
            CatTransform(3),
        ])

        a = resize_transform(np.ones((1, 512, 512)))

        self.assertEqual(a.shape[0], 3)
        self.assertEqual(a.shape[1], 299)
        self.assertEqual(a.shape[2], 299)
