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

import argparse
import math
import os
from multiprocessing import Process
from threading import Thread

import cv2
from PIL import Image

from EyesFinder import EyesFinder
from FaceCascading import FaceCascadingDlib
from ImageCorrection import ImageCorrection


class FacePreparation:
    def __init__(self):
        pass

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

    @staticmethod
    def CropFace(pilImage, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
        # calculate offsets in original image
        offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
        offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
        # distance between them
        dist = FacePreparation.Distance(eye_left, eye_right)
        # calculate the reference eye-width
        reference = dest_sz[0] - 2.0 * offset_h
        # scale factor
        scale = float(dist) / float(reference)

        # rotate image by eyes positions
        image = FacePreparation.RotateFace(pilImage, eye_left, eye_right)

        # crop the rotated image
        crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
        crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
        image = image.crop(
            (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
        # resize it
        image = image.resize(dest_sz, Image.ANTIALIAS)
        return image

    @staticmethod
    def RotateFace(image, eye_left, eye_right, center=None):
        if center is None:
            center = eye_left
        # get the direction
        eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
        # calc rotation angle in radians
        rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
        # rotate original around the left eye
        return FacePreparation.ScaleRotateTranslate(image, center=center, angle=rotation)


class FacePreparationDlib:
    eyesFinder = EyesFinder()
    faceCascading = FaceCascadingDlib()

    def __init__(self):
        pass

    def run(self, args):
        print(args)

        eyes = args['eyes']
        verboseOnly = args['verboseonly']
        offset = args['offset']
        size = args['size']
        fullface = args['fullface']
        if args['image'] is not None:
            for path in args['image']:
                self.__doReadFromFilePath(path,
                                          eyes=eyes, verboseOnly=verboseOnly,
                                          offset=offset, size=size, fullface=fullface)
        elif args['batch'] is not None:
            filePaths = [os.path.join(args['batch'], filename)
                         for filename
                         in os.listdir(args['batch'])
                         if os.path.isfile(os.path.join(args['batch'], filename))]
            for p in filePaths:
                self.__doReadFromFilePath(p,
                                          verboseOnly=verboseOnly, offset=offset,
                                          size=size, fullface=fullface)
        else:
            ap.print_help()
            exit()

    def __doReadFromFilePath(self, path, offset=0.2, size=200, eyes=None,
                             verboseOnly=False, fullface=False, output_folder='face_who'):
        gray = cv2.imread(path)
        if len(gray.shape) > 2:  # color
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        gray = ImageCorrection.normalizeCv2Mat(gray)

        height, width = gray.shape
        if height > 900 and width > 900:
            if height > width:
                height = int(900.0 / width * height)
                width = 900
            else:
                width = int(900.0 / height * width)
                height = 900
            gray = cv2.resize(gray, (width, height))

        # custom eyes coordinations
        if eyes is not None:
            eyesCoords = [int(x) for x in eyes.split(",")]
            eyeLeft = eyesCoords[:2]
            eyeRight = eyesCoords[-2:]
            gray_face = gray
        else:
            if fullface:
                gray_face = gray
            else:
                gray_face = self.faceCascading.detect_face_crop_frame(gray)
                if len(gray_face) == 0:
                    print("no face detected --> full?")
                    gray_face = gray
                elif len(gray_face) > 1:
                    print("more than 1 face detected")
                    exit(1)
                else:
                    gray_face = gray_face[0].copy()
            gray_face = ImageCorrection.equalize_pil_from_cvmat(gray_face)
            eyeLeft, eyeRight = self.eyesFinder.do_find(gray_face)

        if verboseOnly:
            exit(0)

        cropped = FacePreparation.CropFace(ImageCorrection.cv2MatToPilIm(gray_face),
                                           eye_left=eyeLeft, eye_right=eyeRight,
                                           offset_pct=(offset, offset), dest_sz=(size, size))
        path = "./%s/%s.jpg" % (output_folder, '.'.join(os.path.basename(path).split('.')[:-1]))
        print(path)
        cropped.save(path)

    def run_no_wait(self, path, folder):
        t = Process(target=self.run, args={
            {
                'image': path,
                'offset': 0.2,
                'size': 200,
                'folder': folder
            }
        })
        t.daemon = False
        t.start()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    gp = ap.add_mutually_exclusive_group()
    gp.add_argument("-b", "--batch", metavar='BATCH_PATH',
                    help='path of the batch job folder to be processed\n'
                         'for example:"./face/{name}/ok_{code}.*"')
    gp.add_argument("-i", "--image", metavar='img', nargs='+', help='path of the image(s) to be processed')
    ap.add_argument("-o", "--offset", type=float, default=0.2,
                    help="percent of the image you want to keep next to the eyes")
    ap.add_argument("-s", "--size", type=int, default=200, help="width and height of the output image")
    ap.add_argument("-e", "--eyes", metavar='EYES_COORD',
                    help='position of eyes for custom cropping (format: xL,yL,xR,yR)\n'
                         'for example:"100,129,143,128"\n'
                         '(for -i only)')
    ap.add_argument("-ff", "--fullface", action='store_true', help='use full face to find eyes')
    ap.add_argument("-vo", "--verboseonly", action='store_true', help='only output the eye positions info')
    args = vars(ap.parse_args())
    FacePreparationDlib().run(args)
