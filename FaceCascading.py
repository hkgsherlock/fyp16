import cv2
import numpy as np


class FaceCascadingOpencvHaar:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    def detect_face(self, frame):
        # TODO: make parameters be settable via ctor
        faceSizeMin = int(min(frame.shape[0], frame.shape[1]) * 0.05)
        rects = self.face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(faceSizeMin, faceSizeMin),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    def detect_face_crop_frame(self, frame, pos=None):
        if pos is None:
            pos = self.detect_face(frame)
        return [frame[yA:yB, xA:xB] for (xA, yA, xB, yB) in pos]


class FaceCascadingOpencvLbp:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

    def detect_face(self, frame):
        # TODO: make parameters be settable via ctor
        faceSizeMin = int(min(frame.shape[0], frame.shape[1]) * 0.05)
        rects = self.face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(faceSizeMin, faceSizeMin),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    def detect_face_crop_frame(self, frame, pos=None):
        if pos is None:
            pos = self.detect_face(frame)
        return [frame[yA:yB, xA:xB] for (xA, yA, xB, yB) in pos]


class FaceCascadingDlib:
    def __init__(self):
        import dlib
        self.__detector = dlib.get_frontal_face_detector()

    def detect_face(self, frame):
        return np.array([[x, y, x + w, y + h] for (x, y, w, h) in self.detect_face_dlib(frame)])

    def detect_face_dlib(self, frame):
        # dlib_rects, scores, idx = self.__detector.run(frame, 1, -1)
        dlib_rects = self.__detector(frame, 1)
        cv_rects = []
        from imutils import face_utils
        for (i, rect) in enumerate(dlib_rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            x = max(x, 0)
            y = max(y, 0)
            w = max(w, 0)
            h = max(h, 0)
            cv_rects.append((x, y, w, h))
        return cv_rects

    def detect_face_crop_frame(self, frame, pos=None):
        if pos is None:
            pos = self.detect_face(frame)
        return [frame[yA:yB, xA:xB] for (xA, yA, xB, yB) in pos if yB - yA > 0 and xB - xA > 0]
