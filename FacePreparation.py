# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Philipp Wagner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the author nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys, math

import numpy as np
from PIL import Image
import argparse
import cv2


class FacePreparation:
    def __init__(self):
        pass

    def Distance(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    def ScaleRotateTranslate(self, image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
        if (scale is None) and (center is None):
            return image.rotate(angle=angle, resample=resample)
        nx, ny = x, y = center
        sx = sy = 1.0
        if new_center:
            (nx, ny) = new_center
        if scale:
            (sx, sy) = (scale, scale)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        a = cosine / sx
        b = sine / sx
        c = x - nx * a - ny * b
        d = -sine / sy
        e = cosine / sy
        f = y - nx * d - ny * e
        return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)

    def CropFace(self, image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
        # calculate offsets in original image
        offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
        offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
        # get the direction
        eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
        # calc rotation angle in radians
        rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
        # distance between them
        dist = self.Distance(eye_left, eye_right)
        # calculate the reference eye-width
        reference = dest_sz[0] - 2.0 * offset_h
        # scale factor
        scale = float(dist) / float(reference)
        # rotate original around the left eye
        image = self.ScaleRotateTranslate(image, center=eye_left, angle=rotation)
        # crop the rotated image
        crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
        crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
        image = image.crop(
            (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
        # resize it
        image = image.resize(dest_sz, Image.ANTIALIAS)
        return image

    def detectFaceThenEyes(self, path, faceCascade, eyeCascade, glassesCascade):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face, _ = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face = np.array([[x, y, x + w, y + h] for (x, y, w, h) in face])
        face = face[0]
        x1 = face[0]
        y1 = face[1]
        x2 = face[2]
        y2 = face[3]

        eyes, _ = eyeCascade.detectMultiScale(
            image[x1:y1, x2:y2],
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        eyes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in eyes])

        if eyes.size == 2:
            for (xA, yA, xB, yB) in eyes:
                eyes.append([x1 + int(xA + xB / 2.0), y1 + int(yA + yB / 2.0)])
        else:
            glass, _ = glassesCascade.detectMultiScale(
                image[x1:y1, x2:y2],
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE)
            if glass.size != 2:
                exit()
            for (xA, yA, xB, yB) in glass:
                eyes.append([x1 + int(xA + xB / 2.0), y1 + int(yA + yB / 2.0)])

        if eyes[0][0] > eyes[1][0]:
            eyes = list(reversed(eyes))

        return eyes

    def run(self, args):
        ap = argparse.ArgumentParser()
        ap.add_argument('imgPath', metavar='img', nargs='+', help='path of the image(s) to be processed')
        ap.add_argument("-o", "--offset", type=int, default=0.2,
                        help="percent of the image you want to keep next to the eyes")
        ap.add_argument("-s", "--size", type=int, default=200, help="width and height of the output image")
        args = vars(ap.parse_args())

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        glasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        for path in args['imgPath']:
            leftEye, rightEye = self.detectFaceThenEyes(path, face_cascade, eye_cascade, glasses_cascade)

            image = Image.open(path).convert('L')

            offset = args['offset']
            size = args['size']
            cropped = self.CropFace(image, eye_left=leftEye, eye_right=rightEye, offset_pct=(offset, offset),
                               dest_sz=(size, size))

            cropped.save('_crop.'.join(path.rsplit('.', 1)))  # http://stackoverflow.com/q/2556108/2388501

if __name__ == '__main__':
    facePreparation = FacePreparation()
    facePreparation.run(sys.argv)
