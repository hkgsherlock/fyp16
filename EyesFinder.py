import dlib
from imutils import face_utils
import cv2

class EyesFinder:
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    @classmethod
    def do_find(cls, cv_gray):
        h, w = cv_gray.shape[:2]
        shape = cls.predictor(cv_gray, dlib.rectangle(0, 0, w, h))
        shape = face_utils.shape_to_np(shape)
        l = EyesFinder.average_points([shape[37], shape[40]])
        r = EyesFinder.average_points([shape[44], shape[47]])
        return l, r

    @staticmethod
    def average_points(points):
        xx = 0
        yy = 0
        count = 0.0
        for x, y in points:
            xx += x
            yy += y
            count += 1
        return int(xx / count), int(yy / count)
