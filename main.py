import nibabel as nib
import pydicom
import numpy as np
from pydicom.data import get_testdata_files
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from data_loader import *
from graph_cut import GraphCut
from local_binary_patterns import LocalBinaryPatterns
from thresholding import Thresholding


# Pancreas
# dataset = PancreasDataset()
# images, labels = dataset.load_by_id('0002')

# Liver
dataset = LiverDataset()
images, labels = dataset.load_by_id('3')

for img, label in zip(images, labels):

    transform = LocalBinaryPatterns()
    # transform = GraphCut(img, label, dataset.width, dataset.height, showMask=True)
    # transform = Thresholding()
    transform.fit(img, label)


