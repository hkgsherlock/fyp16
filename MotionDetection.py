import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


class MotionDetection:
    def __init__(self, thresholdLow=30, thresholdHigh=255, minAreaSize=200, boundingBoxPadding=20):
        self.thresholdLow = thresholdLow
        self.thresholdHigh = thresholdHigh
        self.minAreaSize = minAreaSize
        self.boundingBoxPadding = boundingBoxPadding
        self.lastFrame = None

    def putNewFrameAndCheck(self, frame, oldFrame=None):
        if oldFrame is None:
            oldFrame = self.lastFrame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frameToLast = gray.copy()

        if oldFrame is None:
            oldFrame = frameToLast

        frameDelta = cv2.absdiff(oldFrame, gray)

        # input, lower limit, upper limit,
        thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # for boundingboxes that are kind of overlapping, combine
        strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, strEl)

        # v-- python2
        # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # python 3
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # for those contours create bounding boxes (basic)
        boundingBoxes = []

        for c in cnts:
            if cv2.contourArea(c) < self.minAreaSize:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            # tuning margin
            xw = x + w + self.boundingBoxPadding
            yh = y + h + self.boundingBoxPadding
            x -= self.boundingBoxPadding
            y -= self.boundingBoxPadding
            boundingBoxes.append([x, y, xw, yh])
            cv2.rectangle(frame, (x, y), (xw, yh), (0, 0, 255), 2)

        boundingBoxesNew = non_max_suppression(np.array(boundingBoxes))

        # for (x1, y1, x2, y2) in boundingBoxesNew:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return boundingBoxesNew
