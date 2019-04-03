import nibabel as nib
import pydicom
import numpy as np
from pydicom.data import get_testdata_files
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

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
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

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
cv_img2 = cv_img.copy()
cv2.namedWindow('img')

# create trackbars for color change
# cv2.createTrackbar('thresh', 'thresh', 50, 255, lambda x: None)


# Interaction fro grab cut
mask = np.ones((COLUMS, ROWS), np.uint8) * cv2.GC_PR_BGD # Grey
rect = (0,0,COLUMS,ROWS)
drawing = False
mode = None
color_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)


def draw_mask(event, x, y, flags, param):
    global mask, drawing, ix, iy, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mode = 'FG'

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        mode = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        mode = 'BG'

    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
        mode = None

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode == 'BG':
                cv2.circle(mask, (x, y), 5, 0, -1)
                cv2.circle(cv_img2, (x, y), 5, 0, -1)
            elif mode == 'FG':
                cv2.circle(mask, (x, y), 5, 1, -1)
                cv2.circle(cv_img2, (x, y), 5, 255, -1)


def redraw_grab_cut(gc_img):
    global temp1, temp2, mask, label
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    mask, _, __ = cv2.grabCut(gc_img, mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_MASK)
    gp_plot = np.where((mask == 0) | (mask == 2), 0, 255).astype(np.uint8)
    print('[DICE]: score' + str(dice_score(gp_plot, label, 255)))
    cv2.imshow('grab-cut', gp_plot)


def dice_score(X, Y, k):
    X = X.copy()
    Y = Y.copy()
    Y[Y == k] = 1
    X[X == k] = 1

    return np.sum(X[Y == 1]) * 2.0 / (np.sum(X) + np.sum(Y))

cv2.setMouseCallback('img', draw_mask)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('d') or k == ord('D'):
        redraw_grab_cut(color_img)

    # Create a black image, a window and bind the function to window
    img = np.zeros((512, 512, 3), np.uint8)

    # get current positions of four trackbars
    # thresholdValue = cv2.getTrackbarPos("thresh", "thresh")

    # Thresholding techniques
    # ret, th1 = cv2.threshold(cv_img, thresholdValue, 255, cv2.THRESH_BINARY)

    # Canny from threshold
    # edges = cv2.Canny(th1, 100, 200)

    # cv2.imshow('thresh', th1)
    cv2.imshow('img', cv_img2)
    cv2.imshow('label', label)
    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    output = cv2.bitwise_and(cv_img, cv_img, mask=mask2)
    cv2.imshow('output', output)


cv2.destroyAllWindows()

# img = (img * 255).astype(np.uint8)
# label = (label * 255).astype(np.uint8)
# cv2.imshow("Img", img)
# cv2.imshow("label", label)
# cv2.waitKey(0)



