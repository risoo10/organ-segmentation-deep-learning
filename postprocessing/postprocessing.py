import cc3d
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from utils.metrics import dice_score
from vtkplotter import *
import cc3d


DIR = 'C:/RISKO/SKOLA/Dimplomka/3d-viz/'


def get_file(data, id):
    return f'UNET_MODEL_LIVER_AUGMENT-loss=WightedDice-epochs=40-{data}-ID-{id}.nii.gz'

def plot_frontal_sagital():
    plt.figure()
    plt.title('Sagital')
    plt.imshow(label[:, 250, :], cmap="bone")


    plt.figure()
    plt.title('Frontal')
    plt.imshow(label[:, :, 250], cmap="bone")

    plt.show()


def list_connectivities(labels_out):
    max_label = np.max(labels_out)
    for i in range(0, max_label):
        print('Label', i, ' VOLUME:', np.sum(labels_out == i))


def postprocess_connectivity(pred):
    labels_out = cc3d.connected_components(pred.astype(np.int32))
    max_label = np.max(labels_out)
    pred = labels_out == 1
    return pred, max_label


def postprocess():
    # LOAD DATA
    id = 0
    pred = nib.load(DIR + get_file('pred', id)).get_fdata()
    pred = np.round(pred)
    label = nib.load(DIR + get_file('lbl', id)).get_fdata()
    label = np.nan_to_num(label)
    label = np.round(label)

    dice_f = dice_score(pred, label)
    print('Loaded data', pred.shape, 'DICE:', dice_f)

    # CONNECTIVITY
    # pred, max_label = postprocess_connectivity(pred)

    
    dice_f = dice_score(pred, label)
    # print('Connectivity | max:', max_label, 'DICE:', dice_f)
    # list_connectivities(labels_out)

    # CREATE AND SHOW SURFACE
    lbl_v = Volume(label)
    lbl_s = lbl_v.isosurface(threshold=[0, 0.5, 1])
    pred_v = Volume(pred)
    pred_s = pred_v.isosurface(threshold=[0, 0.5, 1])
    # s.alpha(0.5).lw(0.1)

    show(lbl_s, pred_s, N=2, axes=8)
