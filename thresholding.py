from skimage import feature
import cv2 as cv2
from utils import *
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
        font = cv2.FONT_HERSHEY_SIMPLEX
        i = 10

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            threshold = cv2.getTrackbarPos('threshold', 'thresh')

            out_img = (img * 255).astype(np.uint8)
            out_label = (label * 255).astype(np.uint8)

            ret, thresh1 = cv2.threshold(out_img, threshold, 255, cv2.THRESH_BINARY)

            # Otsu's thresholding
            ret2, th2 = cv2.threshold(out_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            score_thresh1 = dice_coef(label, thresh1 / 255)
            cv2.putText(thresh1, f'DSC: {score_thresh1}', (15, 50), font, 1, (255, 0, 0), 1, cv2.LINE_AA)

            score_thresh2 = dice_coef(label, th2 / 255)
            cv2.putText(th2, f'DSC: {score_thresh2}', (15, 50), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
            # print(f'[DICE] score: {score}')

            cv2.imshow("img", out_img)
            cv2.imshow("thresh", thresh1)
            cv2.imshow("th2", th2)
            cv2.imshow("label", out_label)
        cv2.destroyAllWindows()
