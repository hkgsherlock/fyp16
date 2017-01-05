import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

def detect_face(frame):
    (rects, weights) = face_cascade.detectMultiScale(frame, 1.3, 5)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    return [frame[xA:yA, xB:yB] for (xA, yA, xB, yB) in rects]
    # for (xA, yA, xB, yB) in rects:
    #     imgFace = frame[xA:yA, xB:yB]
