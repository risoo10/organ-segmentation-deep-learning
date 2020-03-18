import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np


def load_as_npy(path):
    data = nib.load(path).get_fdata()
    data = np.rollaxis(data, 2, 0)
    return data


def load_lits_files_from(path):
    sets = os.listdir(path)
    files = []
    for folder in sets:
        folder_files = os.listdir(os.path.join(path, folder))
        files.extend(
            list(map(lambda x: os.path.join(folder, x), folder_files)))

    volumes = list(filter(lambda x: 'volume' in x, files))
    volumes.sort()

    segmentations = list(filter(lambda x: 'segmentation' in x, files))
    segmentations.sort()

    print('volumes:', len(volumes), '| exmp.',  volumes[:5])
    print('segmentations:', len(segmentations), '| exmp.',  segmentations[:5])

    return volumes, segmentations


def plot_slice(volume, segm, slice_index):
    slice = np.index_exp[slice_index, :, :]
    print(volume.shape)
    print('seg [min, max]', segm[slice].min(),
          segm[slice].max())
    plt.figure(figsize=(15, 15))
    plt.subplot('121')
    plt.title('Source')
    plt.imshow(volume[slice], cmap="bone",)
    plt.subplot('122')
    plt.title('Segmentation')
    seg = segm[slice]
    plt.imshow(seg == 1, cmap="tab20")