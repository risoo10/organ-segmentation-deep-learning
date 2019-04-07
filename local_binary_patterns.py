from skimage import feature
import cv2.cv2 as cv2

from utils import write_mask_contour
import numpy as np


class LocalBinaryPatterns:
    def __init__(self):
        pass

    def fit(self, img, label):
        lbp = feature.local_binary_pattern(img.copy(), 8 * 2, 2, 'uniform')
        lbp = (1 - lbp).astype(np.uint8)
        # ret2, lbp = cv2.threshold(lbp, np.min(lbp), np.max(lbp), cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # lbp = cv2.erode(lbp, cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3)))
        # lbp = cv2.dilate(lbp, cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3)))

        print(f'LBP shape: {lbp.shape}, min:{np.min(lbp)}, max: {np.max(lbp)}')
        img = (img * 255).astype(np.uint8)
        label = (label * 255).astype(np.uint8)
        out = write_mask_contour(img, label)
        cv2.imshow("img", out)
        cv2.imshow("lbp", lbp)
        cv2.imshow("label", label)
        cv2.waitKey(0)
