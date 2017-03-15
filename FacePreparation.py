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

from imutils.object_detection import non_max_suppression


class FacePreparation:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        self.glasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    @staticmethod
    def Distance(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
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

    @staticmethod
    def detectFaceThenEyes(path, faceCascade, eyeCascade, glassesCascade):
        print(path)

        image = cv2.imread(path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_coord = FacePreparation.detectFace(gray, faceCascade)

        if len(face_coord) == 0:
            print("no detected faces, force full face eyes scan")

            # workaround
            hh, ww = gray.shape[:2]
            xx = 0
            yy = 0

        # debug
        drawFrame = image.copy()
        for (xA, yA, xB, yB) in face_coord:
            cv2.rectangle(drawFrame, (xA, yA), (xB, yB), (0, 0, 255), 2)
            cv2.imshow("test", drawFrame)
            cv2.waitKey()

            xx, yy, ww, hh = face_coord[0]

        # debug
        # cv2.imshow("test", gray[yy:hh, xx:ww])
        # cv2.waitKey()

        eyes = eyeCascade.detectMultiScale(
            gray[yy:hh, xx:ww],
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(5, 5),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        eyes = np.array([[x + xx, y + yy, x + xx + w, y + yy + h] for (x, y, w, h) in eyes])

        eyes = non_max_suppression(eyes, probs=None, overlapThresh=0.65)

        # debug
        for (xA, yA, xB, yB) in eyes:
            cv2.rectangle(drawFrame, (xA, yA), (xB, yB), (0, 0, 255), 2)
            cv2.imshow("test", drawFrame)
            cv2.waitKey()

        out_eye = []

        print("eyes len = {}".format(len(eyes)))

        if len(eyes) >= 2:
            for (xA, yA, xB, yB) in eyes:
                eyeXPos = int((xA + xB) / 2.0)
                eyeYPos = int((yA + yB) / 2.0)

                # debug print
                print("{}, {}".format(eyeXPos, eyeYPos))

                out_eye.append([eyeXPos, eyeYPos])
        else:
            glass = glassesCascade.detectMultiScale(
                image[xx:yy, ww:hh],
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(6, 6),
                flags=cv2.CASCADE_SCALE_IMAGE)
            if len(glass) != 2:
                print("glass size 2 != {}".format(len(glass)))
                exit()
            for (xA, yA, xB, yB) in glass:
                eyeRight = [int(xA + xB / 2.0), int(yA + yB / 2.0)]

        if out_eye[0][0] > out_eye[1][0]:
            return out_eye[1], out_eye[0]

        return out_eye[0], out_eye[1]

    @staticmethod
    def detectFace(img, faceCascade):
        face = faceCascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        face = np.array([[x, y, x + w, y + h] for (x, y, w, h) in face])
        print(face)
        return face

    @staticmethod
    def equalize(im):
        layer = 1
        pix = im.getpixel((0, 0))
        if isinstance(pix, tuple):
            layer = len(pix)

        h = im.convert("L").histogram()
        lut = []
        for b in range(0, len(h), 256):
            # step size
            import operator
            step = reduce(operator.add, h[b:b + 256]) / 255
            # create equalization lookup table
            n = 0
            for i in range(256):
                lut.append(n / step)
                n = n + h[i + b]
        # map image through lookup table
        return im.point(lut * layer)

    def run(self):
        ap = argparse.ArgumentParser()
        gp = ap.add_mutually_exclusive_group()
        gp.add_argument("-b", "--batch", metavar='BATCH_PATH',
                        help='path of the batch job folder to be processed\n'
                             'for example:"./face/{name}/ok_{code}.*"')
        gp.add_argument("-i", "--image", metavar='img', nargs='+', help='path of the image(s) to be processed')
        ap.add_argument("-o", "--offset", type=int, default=0.2,
                        help="percent of the image you want to keep next to the eyes")
        ap.add_argument("-s", "--size", type=int, default=200, help="width and height of the output image")
        ap.add_argument("-e", "--eyes", metavar='EYES_COORD',
                        help='position of eyes for custom cropping (format: xL,yL,xR,yR)\n'
                             'for example:"100,129,143,128"')
        args = vars(ap.parse_args())

        print(args)

        if args['image'] is not None:
            for path in args['image']:
                # custom eyes coordinations
                if args['eyes'] is not None:
                    eyesCoords = [int(x) for x in args['eyes'].split(",")]
                    eyeLeft = eyesCoords[:2]
                    eyeRight = eyesCoords[-2:]
                else:
                    eyeLeft, eyeRight = self.detectFaceThenEyes(path, self.face_cascade, self.eye_cascade,
                                                                self.glasses_cascade)

                image = Image.open(path)
                image = image.convert('L')
                image = self.equalize(image)

                offset = args['offset']
                size = args['size']
                cropped = self.CropFace(image, eye_left=eyeLeft, eye_right=eyeRight, offset_pct=(offset, offset),
                                        dest_sz=(size, size))
                newFileName = '.'.join(path.split('.')[:-1])
                cropped.save('{}_ok.jpg'.format(newFileName))  # http://stackoverflow.com/q/2556108/2388501
        elif args['batch'] is not None:
            raise
        else:
            ap.print_help()
            exit()


if __name__ == '__main__':
    FacePreparation().run()
