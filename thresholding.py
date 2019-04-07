from skimage import feature
import cv2.cv2 as cv2

from utils import write_mask_contour
import numpy as np

def nothing(x):
    pass

class Thresholding:
    def __init__(self):
        pass

    def fit(self, img, label):
        cv2.namedWindow("thresh")

        cv2.createTrackbar('threshold', 'thresh', 0, 255, nothing)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            threshold = cv2.getTrackbarPos('threshold', 'thresh')

            out_img = (img * 255).astype(np.uint8)
            out_label = (label * 255).astype(np.uint8)

            ret, thresh1 = cv2.threshold(out_img, threshold, 255, cv2.THRESH_BINARY)

            cv2.imshow("img", out_img)
            cv2.imshow("thresh", thresh1)
            cv2.imshow("label", out_label)
        cv2.destroyAllWindows()
