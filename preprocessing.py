import nibabel as nib
import pydicom
import numpy as np
from pydicom.data import get_testdata_files
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv2

LIVER_DIR = 'C:\RISKO\SKOLA\Dimplomka\Challanges\CHAOS\Data\CT_data_batch - COMBINED 1 and 2\CT_data_batch1'
PANCREAS_DIR = './data/liver/6'

ROWS = 512
COLUMS = 512

# filename = 'data/oranges.jpg'
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image', img)
# cv2.waitKey(0)

# Load Data from pydicom
slice_number = '53'
is_nibabel = False
file = 'i00' + slice_number + ',0000b.dcm'
label_file = 'truth/liver_GT_0' + slice_number + '.png'
ds = pydicom.dcmread(os.path.join(PANCREAS_DIR, file))
img = ds.pixel_array.astype(np.int16)

# Load Label data
if is_nibabel:
    labels = nib.load(os.path.join(PANCREAS_DIR, label_file))
    labels = labels.get_fdata()
    label = labels[:, :, int(slice_number)]

else:
    label = cv2.imread(os.path.join(PANCREAS_DIR, label_file))

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

# img = np.rot90(np.rot90(img))
# label[label == 0] = np.nan
# label = np.flip(np.rot90(np.rot90(np.rot90(label))), axis=0)

# Plot images
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3, 5))
# axes[0].set_title("Data")
# axes[0].imshow(img, cmap="bone")
# axes[1].set_title("Label")
# axes[1].imshow(label, cmap="bone")
# plt.show()

# Add sliders for thresholding
cv_img = (img * 255).astype(np.uint8)
cv2.namedWindow('thresh')

# create trackbars for color change
cv2.createTrackbar('thresh', 'thresh', 50, 255, lambda x: None)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    thresholdValue = cv2.getTrackbarPos("thresh", "thresh")
    ret, th1 = cv2.threshold(cv_img, thresholdValue, 255, cv2.THRESH_BINARY)

    # Canny from threshold
    edges = cv2.Canny(th1, 100, 200)

    cv2.imshow('thresh', th1)
    cv2.imshow('canny', edges)
    cv2.imshow('img', cv_img)
    cv2.imshow('label', label)


cv2.destroyAllWindows()

#



# img = (img * 255).astype(np.uint8)
# label = (label * 255).astype(np.uint8)
# cv2.imshow("Img", img)
# cv2.imshow("label", label)
# cv2.waitKey(0)



