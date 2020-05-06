import torch
from torchvision.transforms import Compose, RandomApply, RandomAffine, ToPILImage, ToTensor
from inception.transforms import ReshapeTransform, CustomResize, CatTransform
import albumentations as alb
from albumentations.torch import ToTensor as AlbumToTensor 
import cv2

PATIENTS, SAMPLES, WIDTH, HEIGHT = 131, 58638, 512, 512

drive_dir = '/content/drive/My Drive/DP'

resize_transform = Compose([
    torch.from_numpy,
    ReshapeTransform((1, 1, WIDTH, HEIGHT)),
    CustomResize(size=(299, 299)),
    ReshapeTransform((1, 299, 299)),
    CatTransform(3),
])


augmentation = alb.Compose([
    alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.5),
])