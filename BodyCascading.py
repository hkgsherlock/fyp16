import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


class BodyCascading:
    def __init__(self):
        self.body_cascade = cv2.HOGDescriptor()
        self.body_cascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.body_cascade_haar = cv2.CascadeClassifier('haarcascade_upperbody.xml')

    def detect(self, frameGray, padding=(8, 8)):
        # TODO: make parameters be settable via ctor
        # detect people in the image
        (rects, weights) = self.body_cascade.detectMultiScale(frameGray, winStride=(4, 4), padding=padding, scale=1.03)

        # convert array type
        rects[:, 2:] = np.add(rects[:, :2], rects[:, 2:])  # (x, y, w, h) --> (x1, y1, x2, y2)
        # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

        # use non_max_suppression to merge unimportant boundingboxes
        rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        return rects  # for (xA, yA, xB, yB) in rects

    def detect_haar_upper_body(self, frameGray, padding=8):
        # TODO: make parameters be settable via ctor
        # detect people in the image

        # hog
        # (rects, weights) = self.body_cascade.detectMultiScale(
        #     frameGray, winStride=(4, 4), padding=(padding, padding), scale=1.03)

        # haar
        faceSizeMin = int(min(frameGray.shape[0], frameGray.shape[1]) * 0.05)
        rects = self.body_cascade_haar.detectMultiScale(
            frameGray,
            scaleFactor=1.1,
            minNeighbors=9,
            minSize=(faceSizeMin, faceSizeMin),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # convert array type
        # hog
        # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        # haar
        rects = np.add(rects, np.array([-padding, -padding, padding, padding]))
        # rects = np.array([[x - padding, y - padding, x + w + padding, y + h + padding] for (x, y, w, h) in rects])

        # use non_max_suppression to merge unimportant boundingboxes
        rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        return rects  # for (xA, yA, xB, yB) in rects
