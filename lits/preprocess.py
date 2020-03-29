import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
import tqdm
import re

from lits.helpers import load_as_npy, plot_slice, find_target_index 
from computer_vision.data_loader import normalize
from lits.lits_dataset import LitsDataSet
from computer_vision.registration import RigidRegistration
from utils.constants import *


def preprocess_lits(vol, segs, REG_TARGET, FILENAME = 'lits-set.h5'):
    try:
        rig_reg = RigidRegistration(REG_TARGET, 0)
        rig_reg.set_source(REG_TARGET)

        lits_dset = LitsDataSet(FILENAME)
        lits_dset.load_write()

        start_index = 0

        for pat_ind, (volume, segm) in tqdm.tqdm(enumerate(zip(vol, segs))):
            vol_num = re.findall(r'.*-(\d+).nii', volume)[0]
            seg_num = re.findall(r'.*-(\d+).nii', segm)[0]
            assert vol_num == seg_num
            id = int(vol_num)

            print('Procesing id:', id)

            # Load
            x = load_as_npy(os.path.join(f'{drive_dir}/LiTS', volume))
            y = load_as_npy(os.path.join(f'{drive_dir}/LiTS', segm))

            # Init
            patient_samples = x.shape[0]
            end_index = start_index + patient_samples

            print('Samples:', patient_samples)

            # Convert and normalize
            y = y != 0      # ignore tumor segmentation
            y = y.astype('int8')
            x = normalize(x)

            # Rotate
            repetitions = 1
            y = rotate90(y, repetitions)
            x = rotate90(x, repetitions)

            # Register
            x, y = register(rig_reg, x, y)
            x = x.astype(np.float16)
            print('D-type | x', x.shape, x.dtype, '| y', y.shape, y.dtype)

            # Save
            lits_dset.save(x, y, start_index, end_index, id)

            start_index = end_index

    finally:
        lits_dset.close()


def rotate90(x, times):
    return np.rot90(x, times, [1, 2])


def load_target():
    path = f'{drive_dir}/LiTS/Training Batch 1/'
    x = load_as_npy(f'{path}volume-4.nii')
    y = load_as_npy(f'{path}segmentation-4.nii')
    y = y != 0
    y = y.astype('int8')
    x = normalize(x)
    target_ind = find_target_index(y)

    x = rotate90(x, 1)
    y = rotate90(y, 1)
    print(target_ind)
    plot_slice(x.astype(np.float32), y, np.index_exp[target_ind])
    return x[target_ind]


def register(reg, volume, mask):
    target_ind = find_target_index(mask)
    print('Start proces: Rigid registration',
          volume.shape, reg.source.shape)
    reg.fit_transform(volume[target_ind])
    patient_samples = volume.shape[0]
    for index in tqdm.tqdm(range(patient_samples)):
        volume[index] = reg.transform_single(volume[index])
        mask[index] = reg.transform_single(mask[index])

    return volume, mask
