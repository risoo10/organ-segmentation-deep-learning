import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


def distance_transform_weight(x):
    if x.max() == 0:
        return np.ones(x.shape) * 2
    else:
        smooth = 0.0000001
        x = 1 - x
        xD = x * 255
        xD = xD.astype(np.uint8)
        dt = cv2.distanceTransform(xD, cv2.DIST_L2, 3)
        dt = (dt / 255)
        dt = dt ** 2
        dt = (dt + smooth / dt.max() + smooth)
        dt[x.astype(np.int)] = 0
        return dt


def tensor_equals(a, b):
    return torch.all(torch.eq(a, b))


class CTDataset(Dataset):
    def __init__(self, x, y, ind, weights=True, transform=None):
        self.x = x
        self.y = y
        self.ind = ind
        self.weights = True
        self.transform = transform
        self.out_transform = transforms.ToTensor()

    def __len__(self,):
        return self.ind.shape[0]

    def __getitem__(self, idx):
        x = self.x[self.ind[idx]].astype(np.float32)
        y = self.y[self.ind[idx]].astype(np.float32)

        if self.transform:
            aug = self.transform(image=x, mask=y)
            x = np.nan_to_num(aug['image'])
            y = np.nan_to_num(aug['mask'])
        else:
            x = np.nan_to_num(x)
            y = np.nan_to_num(y)

        if self.weights:
            weight = distance_transform_weight(y).astype(np.float32)
            weight = np.nan_to_num(weight)
            return self.out_transform(x), self.out_transform(y), self.out_transform(weight)
        else:
            return self.out_transform(x), self.out_transform(y), None


def plot_slice(x, y):
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(np.nan_to_num(x), cmap="bone")
    plt.subplot(1, 2, 2)
    plt.imshow(np.nan_to_num(y), cmap="bone")
    plt.show()


smooth = 0.000001


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def write_mask_contour(img, label):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    label = label.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 10:
            cv2.drawContours(img, contours, i, (0, 0, 255))
    return img


def export_video(images, labels, width, height, filename):
    frame_width = width
    frame_height = height
    out = cv2.VideoWriter(f'{filename}.mp4', cv2.VideoWriter_fourcc('H', '2', '6', '4'), 20,
                          (frame_width, frame_height))
    assert images.shape == labels.shape
    for index in range(images.shape[0]):
        slice = images[index]
        img = (slice * 255).astype(np.uint8)
        # Find and draw label as contour
        label = (labels[index] * 255).astype(np.uint8)
        img = write_mask_contour(img, label)
        out.write(img)
    out.release()


def export_png(array, output_dir, filename):
    # Export to PNG
    for lbl_id in range(array.shape[0]):
        path = os.path.join(output_dir, filename, str(lbl_id)) + '.png'
        lbl_img = (array[lbl_id] * 255).astype(np.uint8)
        cv2.imwrite(path, lbl_img)


def tensorToNumpy(tensor, round=False):
  if tensor.is_cuda:
    tensor = tensor.detach().cpu()
  tensor = tensor.numpy()
  if round :
      tensor = np.round(tensor)
  return tensor


def printMetrics(metrics, scores, end='\n', title='Metrics'):
  text = f'{title}: '
  for i, metric in enumerate(metrics):
    text = f'{text}{metric.name}={np.round(scores[i], 5)}, '
    
  print(text, end=end)

