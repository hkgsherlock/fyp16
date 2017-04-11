import os

import cv2
import dlib
import numpy as np
from PIL import Image


class FaceRecognisingOpencv:
    def __init__(self, threshold=128.0, createMethod=cv2.face.createLBPHFaceRecognizer, prepareImmediately=True):
        self.__model = createMethod(threshold=threshold)
        self.__labels = []
        if prepareImmediately:
            self.prepare()

    def __get_images_and_labels(self, path):
        images = []
        labels = []

        # ./face/{name}/{ok_code}.*
        for dir in [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]:
            dirFull = os.path.join(path, dir)
            id = len(self.__labels)
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
                self.__labels.append((id, dir))
        # return the images list and labels list
        return images, labels

    def prepare(self):
        images, labels = self.__get_images_and_labels('./face')
        self.__model.train(images, np.array(labels))
        for i, s in self.__labels:
            self.__model.setLabelInfo(i, s)

    def predict(self, image):
        image = cv2.resize(image, (200, 200))
        labelId, confidence = self.__model.predict(image)
        if labelId == -1:
            return -1, confidence
        return self.getLabelFromId(labelId), confidence

    def getLabelFromId(self, index):
        return self.__model.getLabelInfo(index)
        # return self.labels[index]

    def save(self):
        if not os.path.exists("./rec_prof") or not os.path.isdir("./rec_prof"):
            os.mkdir("./rec_prof")
        self.__model.save("./rec_prof/prof.xml")

    def load(self, filename):
        self.__model.load("./rec_prof/{}".format(filename))

    def setThreshold(self, value):
        self.__model.setThreshold(value)


# class FaceRecognisingDlib:
#     __facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
#     __sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
#     def __init__(self, prepareImmediately=True):
#         self.__descriptors = []
#         if prepareImmediately:
#             self.prepare()
#
#     def __get_images_and_labels(self, path):
#         images = []
#         labels = []
#
#         # ./face/{name}/{ok_code}.*
#         for dir in [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]:
#             dirFull = os.path.join(path, dir)
#             subfiles = [o for o in os.listdir(dirFull) if os.path.isfile(os.path.join(dirFull, o))]
#             for image_fname in subfiles:
#                 # # remove in production
#                 # if not image_fname.split('.')[0].endswith('_ok'):
#                 #     continue
#                 image_path = os.path.join(dirFull, image_fname)
#                 image_pil = Image.open(image_path).convert('L')
#                 image = np.array(image_pil, 'uint8')
#                 images.append(image)
#                 labels.append(id)
#         # return the images list and labels list
#         return images, labels
#
#     def prepare(self):
#         for img, lbl in self.__get_images_and_labels('./face'):
#             vec = self.__getDescriptorValuesFullImage(img)
#             self.__descriptors.append(lbl, vec)
#
#     def predict(self, image):
#         dist = []
#         vec = self.__getDescriptorValuesFullImage(image)
#         for i in descriptors:
#             dist_ = numpy.linalg.norm(i - d_test)
#             dist.append(dist_)
#
#     def save(self):
#         if not os.path.exists("./rec_prof") or not os.path.isdir("./rec_prof"):
#             os.mkdir("./rec_prof")
#         pass
#
#     def load(self, filename):
#         pass
#
#     def setThreshold(self, value):
#         pass
#
#     @classmethod
#     def __getDescriptorValuesFullImage(cls, img):
#         height, width = img.shape
#         shape = cls.__sp(img, dlib.rectangle(0, 0, width, height))
#         face_descriptor = cls.__facerec.compute_face_descriptor(img, shape)
#         return np.array(face_descriptor)
