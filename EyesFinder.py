import dlib
import numpy as np
from imutils import face_utils


class EyesFinder:
    def __init__(self):
        pass

    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    @classmethod
    def do_find(cls, cv_gray):
        h, w = cv_gray.shape[:2]
        shape = cls.predictor(cv_gray, dlib.rectangle(0, 0, w, h))
        shape = face_utils.shape_to_np(shape)
        l = tuple(EyesFinder.average_points(shape[36:42]))
        r = tuple(EyesFinder.average_points(shape[42:48]))
        return l, r

    @staticmethod
    def average_points(points):
        return np.array(np.average(points, axis=0), 'int')
