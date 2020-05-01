import numpy as np
import cv2.cv2 as cv2
from livseg_utils import write_mask_contour, dice_coef


# This code was modified from the official OPENCV github page for samples
# with source at https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py
class GraphCut:
    def __init__(self, image, label, columns, rows, show_mask=False):
        assert image.shape == label.shape
        self.image = image
        self.label = label
        self.cv_img = (image * 255).astype(np.uint8)
        self.cv_img2 = self.cv_img.copy()
        self.columns = columns
        self.rows = rows

        # Interaction for grab cut
        self.mask = np.ones((self.columns, self.rows), np.uint8) * cv2.GC_PR_BGD  # Grey
        self.rect = (0, 0, self.columns, self.rows)
        self.drawing = False
        self.mode = None
        self.show_mask = show_mask
        self.score = None

        if self.show_mask:
            self.cv_img2 = write_mask_contour(self.cv_img2, label)

    def draw_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mode = 'FG'

        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.drawing = False
            self.mode = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mode = 'BG'

        if event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.mode == 'BG':
                cv2.circle(self.mask, (x, y), 5, 0, -1)
                cv2.circle(self.cv_img2, (x, y), 5, 0, -1)
            elif self.mode == 'FG':
                cv2.circle(self.mask, (x, y), 5, 1, -1)
                cv2.circle(self.cv_img2, (x, y), 5, 255, -1)

    def redraw_grab_cut(self, gc_img):
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        self.mask, _, __ = cv2.grabCut(gc_img, self.mask, self.rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_MASK)
        gp_plot = np.where((self.mask == 0) | (self.mask == 2), 0, 255).astype(np.uint8)
        self.score = dice_coef(self.label, gp_plot / 255)
        print(f'[DSC]: score f {self.score}')
        cv2.imshow('grab-cut', gp_plot)

    def fit(self, image, label):
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.draw_mask)
        color_img = cv2.cvtColor(self.cv_img, cv2.COLOR_GRAY2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord('d') or k == ord('D'):
                self.redraw_grab_cut(color_img)

            mask2 = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype('uint8')
            out = self.cv_img2.copy()
            cv2.putText(out, f'DSC: {self.score}', (15, 50), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
            output = cv2.bitwise_and(self.cv_img, self.cv_img, mask=mask2)
            cv_label = (self.label * 255).astype(np.uint8)

            cv2.imshow('img', out)
            cv2.imshow('label', cv_label)
            cv2.imshow('output', output)

        cv2.destroyAllWindows()
