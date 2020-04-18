import torch
import torch.nn.functional as nn
from torch.utils.data import Dataset, DataLoader


class CustomResize:
  def __init__(self, size):
    self.size = size

  def __call__(self, x):
    return nn.interpolate(x, self.size, mode="bilinear")

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

class CatTransform:
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, x):
      out = [x for i in range(self.channels)]
      return torch.cat(out, 0)
        