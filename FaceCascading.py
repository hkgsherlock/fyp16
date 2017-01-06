import cv2
import numpy as np


class FaceCascading:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

    def detect_face(self, frame):
        # TODO: make parameters be settable via ctor
        (rects, weights) = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        return np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    def detect_face_crop_frame(self, frame, pos=None):
        if pos is None:
            pos = self.detect_face(frame)
        return [frame[xA:yA, xB:yB] for (xA, yA, xB, yB) in pos]

# for (xA, yA, xB, yB) in rects:
#     imgFace = frame[xA:yA, xB:yB]
