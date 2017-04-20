from threading import Thread

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from Queue import Queue

from Debugger import DataView


class NoWaitMotionDetection:
    def __init__(self, thresholdLow=50, thresholdHigh=255, minAreaSize=1000, boundingBoxPadding=20, frameSpan=4):
        self.__md = MotionDetection(thresholdLow=thresholdLow,
                                    thresholdHigh=thresholdHigh,
                                    minAreaSize=minAreaSize,
                                    boundingBoxPadding=boundingBoxPadding,
                                    frameSpan=frameSpan)
        self.reading = []
        self.__q_frame = Queue()
        self.__t = Thread(target=self.__t_run)
        self.__t.daemon = True
        self.__t.start()
        self.__i = 0

    def putNewFrameAndCheck(self, frame):
        self.__q_frame.put(frame, block=False)
        # print(self.reading)
        return self.reading

    def __t_run(self):
        while True:
            if not self.__q_frame.empty():
                f = self.__q_frame.get()
                self.reading = self.__md.putNewFrameAndCheck(f)


class MotionDetection:
    def __init__(self, thresholdLow=50, thresholdHigh=255, minAreaSize=1000, boundingBoxPadding=20, frameSpan=4):
        self.frameSpan = frameSpan
        self.thresholdLow = thresholdLow
        self.thresholdHigh = thresholdHigh
        self.minAreaSize = minAreaSize
        self.boundingBoxPadding = boundingBoxPadding
        self.lastFrames = Queue()

    def putNewFrameAndCheck(self, frame, oldFrame=None):
        gray = frame
        if len(gray.shape) != 2:  # not gray
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        if oldFrame is None:
            if self.lastFrames.qsize() < self.frameSpan:
                if self.lastFrames.empty():
                    self.lastFrames.put(gray)
                    return []
                else:
                    oldFrame = self.lastFrames.queue[0]  # peek
            else:
                oldFrame = self.lastFrames.get()

        frameDelta = cv2.absdiff(oldFrame, gray)
        # from Debugger import DataView
        # DataView.show_image_nowait(frameDelta, "delta")

        # input, lower limit, upper limit,
        thresh = cv2.threshold(frameDelta, self.thresholdLow, self.thresholdHigh, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # for boundingboxes that are kind of overlapping, combine
        # strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, strEl)

        # v-- python2
        # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # python 3
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # for those contours create bounding boxes (basic)
        bbRects = np.array([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= self.minAreaSize])
        # print(bbRects)

        if len(bbRects) > 0:
            bbRects[:, 2:] = np.add(bbRects[:, :2], bbRects[:, 2:])  # (x, y, w, h) --> (x1, y1, x2, y2)
            pad = self.boundingBoxPadding
            bbRects = np.add(bbRects, np.array([-pad, -pad, pad, pad]))
            # bbRects = non_max_suppression(np.array(bbRects))

            # print(bbRects)
            bbRects = np.array([np.amin(bbRects[:, :2], axis=0),
                                np.amax(bbRects[:, 2:], axis=0)]).reshape((-1, 4))
            bbRects[bbRects < 0] = 0  # remove negative values

        # for (x1, y1, x2, y2) in boundingBoxesNew:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if oldFrame is not None:
            self.lastFrames.put(gray)

        # print(bbRects)
        return bbRects
