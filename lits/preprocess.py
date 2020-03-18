import matplotlib.pyplot as plt
import os
import nibabel as nibabel
import numpy as np

def load_lits_files(path):
  sets = os.listdir(path)
  files = []
  for folder in sets:
    folder_files = os.listdir(os.path.join(path, folder))
    files.extend(list(map(lambda x: os.path.join(folder, x), folder_files)))

  volumes = list(filter(lambda x: 'volume' in x, files))
  volumes.sort()

  segmentations = list(filter(lambda x: 'segmentation' in x, files))
  segmentations.sort()

  print('volumes:', len(volumes), '| exmp.',  volumes[:5])
  print('segmentations:', len(segmentations), '| exmp.',  segmentations[:5])

  return volumes, segmentations

def plot_slice(volume, segm, slice_index):
  print(volume.shape)
  print('seg [min, max]', segm[:, :, slice_index].min(), segm[:, :, slice_index].max())
  plt.figure(figsize=(15, 15))
  plt.subplot('121')
  plt.title('Source')
  plt.imshow(volume[:, :, slice_index], cmap="bone",)
  plt.subplot('122')
  plt.title('Segmentation')
  seg = segm[:, :, slice_index]
  plt.imshow(seg == 1, cmap="tab20")