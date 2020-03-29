import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np


def find_target_index(segmentation):
    organ_detected = np.array(list(map(lambda x: np.any(x), segmentation)))
    first = np.argmax(organ_detected)
    last = len(organ_detected) - np.argmax(np.flip(organ_detected)) - 1
    target = int((first + last) / 2)
    print('Organ detection', first, last, target)
    return target


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


def plot_slice(volume, segm, slice, figsize=(15, 15)):
    x = volume[slice].astype(np.float32)
    y = segm[slice].astype(np.int16)
    print(volume.shape)
    print('seg [min, max]', y.min(), y.max())
    plt.figure(figsize=figsize)
    plt.subplot('121')
    plt.title('Source')
    plt.imshow(x, cmap="bone",)
    plt.subplot('122')
    plt.title('Segmentation')
    plt.imshow(y, cmap="tab20")


def plot_patient_samples(indexes, dset, scale_factor=2, columns=8):
    length = len(indexes)
    rows = np.ceil(length / columns)
    plt.figure(figsize=(columns * scale_factor, rows * scale_factor))
    for i, index in enumerate(indexes):
        pat_index = dset.slices[index]
        ind = int((pat_index[0] + pat_index[1]) / 2)
        plt.subplot(rows, columns, i + 1)
        plt.title(f'Patient {index}')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(dset.x[ind].astype(np.float32), cmap="bone")
