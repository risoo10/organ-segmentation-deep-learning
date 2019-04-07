import numpy as np
import cv2.cv2 as cv2

from metrics import dice_score
from utils import write_mask_contour


class GraphCut:
    def __init__(self, image, label, COLUMNS, ROWS, showMask=False):
        assert image.shape == label.shape
        self.image = image
        self.label = label
        self.cv_img = (image * 255).astype(np.uint8)
        self.cv_img2 = self.cv_img.copy()
        self.COLUMS = COLUMNS
        self.ROWS = ROWS

        # Interaction for grab cut
        self.mask = np.ones((self.COLUMS, self.ROWS), np.uint8) * cv2.GC_PR_BGD # Grey
        self.rect = (0, 0, self.COLUMS, self.ROWS)
        self.drawing = False
        self.mode = None
        self.showMask = showMask

        if self.showMask:
            self.cv_img2 = write_mask_contour(self.cv_img2, label)


    def draw_mask(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mode = 'FG'

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.mode = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.mode = 'BG'

        elif event == cv2.EVENT_RBUTTONUP:
            self.drawing = False
            self.mode = None

        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
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
        print('[DICE]: score' + str(dice_score(gp_plot, self.label, 255)))
        cv2.imshow('grab-cut', gp_plot)

    def fit(self, image, label):
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.draw_mask)
        color_img = cv2.cvtColor(self.cv_img, cv2.COLOR_GRAY2BGR)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord('d') or k == ord('D'):
                self.redraw_grab_cut(color_img)

            # Create a black image, a window and bind the function to window
            # img = np.zeros((512, 512, 3), np.uint8)

            # get current positions of four trackbars
            # thresholdValue = cv2.getTrackbarPos("thresh", "thresh")

            # Thresholding techniques
            # ret, th1 = cv2.threshold(self.cv_img, thresholdValue, 255, cv2.THRESH_BINARY)

            # Canny from threshold
            # edges = cv2.Canny(th1, 100, 200)

            # cv2.imshow('thresh', th1)
            cv2.imshow('img', self.cv_img2)
            cv_label = (self.label * 255).astype(np.uint8)
            cv2.imshow('label', cv_label)
            mask2 = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype('uint8')
            output = cv2.bitwise_and(self.cv_img, self.cv_img, mask=mask2)
            cv2.imshow('output', output)

        cv2.destroyAllWindows()
