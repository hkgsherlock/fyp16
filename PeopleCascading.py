import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


class PeopleCascading:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        # get the default people detector
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frameGray):
        # TODO: make parameters be settable via ctor
        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(frameGray, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # convert array type
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

        # use non_max_suppression to merge unimportant boundingboxes
        rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        return rects  # for (xA, yA, xB, yB) in rects
