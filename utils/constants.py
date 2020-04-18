import torch
from torchvision.transforms import Compose
from inception.transforms import ReshapeTransform, CustomResize, CatTransform

PATIENTS, SAMPLES, WIDTH, HEIGHT = 131, 58638, 512, 512

drive_dir = '/content/drive/My Drive/DP'

resize_transform = Compose([
        torch.from_numpy,
        ReshapeTransform((1, 1, WIDTH, HEIGHT)),
        CustomResize(size=(299, 299)),
        ReshapeTransform((1, 299, 299)),
        CatTransform(3),
])