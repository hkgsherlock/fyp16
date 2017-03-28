import os

import cv2
import numpy as np
from PIL import Image


class FaceRecognising:
    def __init__(self, threshold=128):
        self.model = cv2.face.createLBPHFaceRecognizer(threshold=threshold)
        self.labels = []
        # self.prepare()

    def get_images_and_labels(self, path):
        images = []
        labels = []

        # ./face/{name}/{ok_code}.*
        for dir in [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]:
            dirFull = os.path.join(path, dir)

            for image_fname in [o for o in os.listdir(dirFull) if os.path.isfile(os.path.join(dirFull, o))]:
                # # remove in production
                # if not image_fname.split('.')[0].endswith('_ok'):
                #     continue
                image_path = os.path.join(path, dir, image_fname)
                image_pil = Image.open(image_path).convert('L')
                image = np.array(image_pil, 'uint8')
                images.append(image)
                labels.append(len(self.labels))
                self.labels.append(dir)
        # return the images list and labels list
        return images, labels

    def prepare(self):
        images, labels = self.get_images_and_labels('./face')
        self.model.train(images, np.array(labels))

    def predict(self, image):
        labelId, confidence = self.model.predict(image)
        if labelId == -1:
            return -1, confidence
        return self.labels[labelId], confidence

    def getLabelFromId(self, index):
        return self.labels[index]
