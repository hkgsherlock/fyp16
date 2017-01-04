import os

import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')


def get_images_and_labels(path):
    images = []
    labels = []

    # ./face/{name}/{ok_code}.*
    for dir in [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]:
        for image_fname in os.path.join(path, dir):
            if not image_fname.startswith("ok_"):
                continue
            image_path = os.path.join(path, dir, image_fname)
            image_pil = Image.open(image_path).convert('L')
            image = np.array(image_pil, 'uint8')
            images.append(image)
            labels.append(dir)
    # return the images list and labels list
    return images, labels


def prepare_faceRecognizer():
    _model = cv2.face.createLBPHFaceRecognizer()
    images, labels = get_images_and_labels('./face')
    # _model.train
    return _model


def detect_face(frame):
    (rects, weights) = face_cascade.detectMultiScale(frame, 1.3, 5)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    return [frame[xA:yA, xB:yB] for (xA, yA, xB, yB) in rects]
    # for (xA, yA, xB, yB) in rects:
    #     imgFace = frame[xA:yA, xB:yB]
    # TODO: what to do after you found a face? write on camera.py
