import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_slice(x, y):
  figure = plt.figure(figsize=(20, 20))
  plt.subplot(1, 2, 1)
  plt.imshow(np.nan_to_num(x), cmap="bone")
  plt.subplot(1, 2, 2)
  plt.imshow(np.nan_to_num(y), cmap="bone")
  plt.show()


smooth = 0.000001
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def write_mask_contour(img, label):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    label = label.astype(np.uint8)
    contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 10:
            cv2.drawContours(img, contours, i, (0, 0, 255))
    return img


def export_video(images, labels, width, height, filename):
    frame_width = width
    frame_height = height
    out = cv2.VideoWriter(f'{filename}.mp4', cv2.VideoWriter_fourcc('H', '2', '6', '4'), 20, (frame_width, frame_height))
    assert images.shape == labels.shape
    for index in range(images.shape[0]):
        slice = images[index]
        img = (slice * 255).astype(np.uint8)
        # Find and draw label as contour
        label = (labels[index] * 255).astype(np.uint8)
        img = write_mask_contour(img, label)
        out.write(img)
    out.release()


def export_png(array, output_dir, filename):
    # Export to PNG
    for lbl_id in range(array.shape[0]):
        path = os.path.join(output_dir, filename, str(lbl_id)) + '.png'
        lbl_img = (array[lbl_id] * 255).astype(np.uint8)
        cv2.imwrite(path, lbl_img)

