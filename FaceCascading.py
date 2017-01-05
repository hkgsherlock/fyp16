import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')


def detect_face(frame):
    (rects, weights) = face_cascade.detectMultiScale(frame, 1.3, 5)
    return np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])


def detect_face_crop_frame(frame, pos=None):
    if pos is None:
        pos = detect_face(frame)
    return [frame[xA:yA, xB:yB] for (xA, yA, xB, yB) in pos]


# for (xA, yA, xB, yB) in rects:
#     imgFace = frame[xA:yA, xB:yB]
