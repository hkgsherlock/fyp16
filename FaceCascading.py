import cv2
import numpy as np


class FaceCascading:
    def __init__(self):
        # self.face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    def detect_face(self, frame):
        # TODO: make parameters be settable via ctor
        faceSizeMin = int(min(frame.shape[0], frame.shape[1]) * 0.1)
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

# for (xA, yA, xB, yB) in rects:
#     imgFace = frame[yA:yB, xA:xB]
