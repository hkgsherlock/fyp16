import os

import cv2
import numpy as np
from PIL import Image


class FaceRecognising:
    def __init__(self, threshold=128, createMethod=cv2.face.createLBPHFaceRecognizer, prepareImmediately=True):
        self.model = createMethod(threshold=threshold)
        self.labels = []
        if prepareImmediately:
            self.prepare()

    def get_images_and_labels(self, path):
        images = []
        labels = []

        # ./face/{name}/{ok_code}.*
        for dir in [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]:
            dirFull = os.path.join(path, dir)
            id = len(self.labels)
            subfiles = [o for o in os.listdir(dirFull) if os.path.isfile(os.path.join(dirFull, o))]
            for image_fname in subfiles:
                # # remove in production
                # if not image_fname.split('.')[0].endswith('_ok'):
                #     continue
                image_path = os.path.join(dirFull, image_fname)
                image_pil = Image.open(image_path).convert('L')
                image = np.array(image_pil, 'uint8')
                images.append(image)
                labels.append(id)
            if len(subfiles) > 0:
                self.labels.append((id, dir))
        # return the images list and labels list
        return images, labels

    def prepare(self):
        images, labels = self.get_images_and_labels('./face')
        self.model.train(images, np.array(labels))
        for i, s in self.labels:
            self.model.setLabelInfo(i, s)

    def predict(self, image):
        labelId, confidence = self.model.predict(image)
        if labelId == -1:
            return -1, confidence
        return self.labels[labelId], confidence

    def getLabelFromId(self, index):
        return self.labels[index]

    def save(self):
        if not os.path.exists("./rec_prof") or not os.path.isdir("./rec_prof"):
            os.mkdir("./rec_prof")
        self.model.save("./rec_prof/prof.xml")

    def load(self, filename):
        self.model.load("./rec_prof/{}".format(filename))

    def setThreshold(self, value):
        self.model.setThreshold(value)
