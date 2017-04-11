import os

import cv2

from BodyCascading import BodyCascading

import argparse

ap = argparse.ArgumentParser()
gp = ap.add_mutually_exclusive_group()
gp.add_argument("-b", "--batch", metavar='BATCH_PATH',
                help='path of the batch job folder to be processed\n'
                     'for example:"./face/{name}/ok_{code}.*"')
gp.add_argument("-i", "--img", metavar='IMAGES', nargs='*', help='image to be processed')
args = vars(ap.parse_args())

files = []
if args['batch'] is not None:
    for o in os.listdir(args['batch']):
        path = os.path.join(args['batch'], o)
        if os.path.isfile(path):
            files.append(path)
else:
    files = args['img']

bc = BodyCascading()

for imgP in files:
    print(imgP)
    im = cv2.imread(imgP)
    height, width = im.shape[:-1]
    if height > 600 and width > 600:
        if height > width:
            height = int(600.0 / width * height)
            width = 600
        else:
            width = int(600.0 / height * width)
            height = 600
        im = cv2.resize(im, (width, height))
    g = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY)
    bb = bc.detect(g)
    # print("detected peds: %d" % len(bb))
    for x1, y1, x2, y2 in bb:
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 5)
    # cv2.imshow("t", im)
    # cv2.waitKey(1)
    cv2.imwrite("./test/output/%s.png" % '.'.join(os.path.basename(imgP).split('.')[:-1]), im)
    # cv2.destroyWindow("t")
