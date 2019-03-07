import nibabel as nib
import pydicom
import numpy as np
from pydicom.data import get_testdata_files
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv2

LIVER_DIR = 'C:\RISKO\SKOLA\Dimplomka\Challanges\CHAOS\Data\CT_data_batch - COMBINED 1 and 2\CT_data_batch1'
PANCREAS_DIR = './data/pancreas/0003-59468/'

ROWS = 512
COLUMS = 512

# filename = 'data/oranges.jpg'
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image', img)
# cv2.waitKey(0)

# Load Data from pydicom
slice_number = '159'
pancreas_file = '000' + slice_number + '.dcm'
pancreas_label_file = 'label0003.nii.gz'
ds = pydicom.dcmread(os.path.join(PANCREAS_DIR, pancreas_file))
img = ds.pixel_array.astype(np.int16)

# Load Label data
labels = nib.load(os.path.join(PANCREAS_DIR, pancreas_label_file))
labels = labels.get_fdata()
label = labels[:, :, int(slice_number)]

# Convert to Hounsfield units (HU)
intercept = ds.RescaleIntercept
slope = ds.RescaleSlope
if slope != 1:
    img = slope * img.astype(np.float64)
    img = img.astype(np.int16)

img += np.int16(intercept)
img = np.array(img, dtype=np.int16)

# Set outside-of-scan pixels to 0
img[img < -2000] = intercept

# Clip only HU of liver and tissues
img = np.clip(img, -100, 300)

# Normalize input
copy = img
min_, max_ = float(np.min(copy)), float(np.max(copy))
img = (copy - min_) / (max_ - min_)

img = np.rot90(np.rot90(img))
label[label == 0] = np.nan
label = np.flip(np.rot90(np.rot90(np.rot90(label))), axis=0)

plt.imshow(img, cmap="bone")
plt.imshow(label, alpha=.25, cmap="spring")
plt.show()


# img = (img * 255).astype(np.uint8)
# label = (label * 255).astype(np.uint8)
# cv2.imshow("Img", img)
# cv2.imshow("label", label)
# cv2.waitKey(0)



