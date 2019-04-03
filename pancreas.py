import nibabel as nib
import pydicom
import numpy as np
from pydicom.data import get_testdata_files
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
import SimpleITK as sitk

LIVER_DIR = 'C:\RISKO\SKOLA\Dimplomka\Challanges\CHAOS\Data\CT_data_batch - COMBINED 1 and 2\CT_data_batch1'
# PANCREAS_DIR = './data/liver/6'
PANCREAS_DIR = 'C:\RISKO\SKOLA\Dimplomka\Challanges\CT-PANCREAS\Pancreas-CT'
PANCREAS_LABELS_DIR = 'C:\RISKO\SKOLA\Dimplomka\Challanges\CT-PANCREAS\PANCREAS-LABELS'
PATIENT_ID = '0068'
PATIENT_DIR = os.path.join(PANCREAS_DIR, f'PANCREAS_{PATIENT_ID}')

ROWS = 512
COLUMS = 512

# filename = 'data/oranges.jpg'
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image', img)
# cv2.waitKey(0)

def export_labels():
    # Export to PNG
    EXPORT_DIR = "PNG_LABELS"
    for lbl_id in range(labels.shape[2]):
        path = os.path.join(PANCREAS_DIR, EXPORT_DIR, str(lbl_id)) + '.png'
        print(lbl_id, 'z', labels.shape[2], ':', path)
        lbl_img = (labels[:, :, lbl_id] * 255).astype(np.uint8)
        cv2.imwrite(path, lbl_img)


# Load Data from pydicom
is_nibabel = True
dir1 = os.listdir(PATIENT_DIR)[0]
dir2 = os.listdir(os.path.join(PATIENT_DIR, dir1))[0]
final_dir = os.path.join(PATIENT_DIR, dir1, dir2)
slices = os.listdir(final_dir)

# Load Label data
if is_nibabel:
    label_file = f'label{PATIENT_ID}.nii.gz'
    path = os.path.join(PANCREAS_LABELS_DIR, label_file)
    volume = sitk.ReadImage(path)
    labels = sitk.GetArrayFromImage(volume).astype(np.float64)
    print(f'Labels shape: {labels.shape}')

output = np.zeros((len(slices) + 100, ROWS, COLUMS))
min_slice = 1000
max_slice = 0
for slice_number, slice_file in enumerate(slices):
    ds = pydicom.dcmread(os.path.join(final_dir, slice_file))
    slice_index = ds.data_element("InstanceNumber").value - 1
    img = ds.pixel_array.astype(np.int16)
    min_slice = slice_index if slice_index < min_slice else min_slice
    max_slice = slice_index if slice_index > max_slice else max_slice
    print(f'Instance number: {slice_index}, Shape: {img.shape}')

    if is_nibabel:
        label = labels[slice_index]
        # labels = nib.load(path)
        # labels = labels.get_fdata()
        # label = labels[:, :, slice_index]
        # # export_labels()

    else:
        label_file = 'truth/liver_GT_0' + slice_number + '.png'
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
    output[slice_index] = img

    # plt.imshow(img, cmap="bone")
    # plt.show()

    # img = (img * 255).astype(np.uint8)
    # label = (label * 255).astype(np.uint8)
    # cv2.imshow("Img", img)
    # cv2.imshow("label", label)
    # label[label != 0] = 100
    #     # label[label == 0] = np.nan
    #     # # img = np.rot90(np.rot90(img))
    #     # plt.title(str(slice_index))
    #     # plt.imshow(img, cmap="bone")
    #     # plt.imshow(label, cmap="hsv", alpha=0.4)
    #     # plt.show()
    #     # cv2.waitKey(0)

print(f'Finished processing, minSlice: {min_slice}, maxSlice: {max_slice}')

# Export video from slices
frame_width = COLUMS
frame_height = ROWS

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(f'plots/pancreas{PATIENT_ID}.mp4', cv2.VideoWriter_fourcc('H', '2', '6', '4'), 20, (frame_width, frame_height))
for index in range(min_slice, max_slice):
    slice = output[index]
    img = (slice * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Find and draw contours
    label = (labels[index] * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 10:
            cv2.drawContours(img, contours, i, (0, 0, 255))
    # cv2.imshow("Slice", img)
    # cv2.waitKey(0)
    out.write(img)
out.release()


