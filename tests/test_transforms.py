import unittest
import torch
import numpy as np
from inception.transforms import CatTransform, CustomResize, ReshapeTransform
from utils.constants import *
import matplotlib.pyplot as plt
from torchvision.transforms import Compose


class TestLitsSet(unittest.TestCase):
    def test_augmentation_transform(self):
        a = np.zeros((512, 512)).astype(np.float32)
        a[150:250, 150:250] = 1
        # a = torch.from_numpy(a)
        aug = augmentation(image=a)
        b = aug["image"]

        print('[ALBUMENTATIONS]', a.shape, b.shape)
        
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(a)
        # plt.subplot(1, 2, 2)
        # plt.imshow(b)
        # plt.show()