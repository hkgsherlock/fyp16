import os

import cv2

from FaceCascading import FaceCascading
from FaceRecognising import FaceRecognising

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

fc = FaceCascading()
fr = FaceRecognising()

eyeCc = cv2.CascadeClassifier('haarcascade_eye.xml')

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
    detected_faces = fc.detect_face(g)
    # print("detected faces: %d" % len(detected_faces))
    for x1, y1, x2, y2 in detected_faces:
        # print("%d,%d,%d,%d" % (x1, y1, x2, y2))
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 5)
        dist = int((x2 - x1) * .05)
        face_g = g[y1:y2, x1:x2]
        # for ex1, ey1, ex2, ey2 in eyeCc.detectMultiScale(
        #     face_g,
        #     scaleFactor=1.1,
        #     minNeighbors=5,
        #     flags=cv2.CASCADE_SCALE_IMAGE
        # ):
        #     # print("%d,%d,%d,%d" % (x1, y1, x2, y2))
        #     cv2.rectangle(im, (ex1 + x1, ey1 + x1), (ex2 + x1, ey2 + x1), (0, 255, 0), 5)
        name, confidence = fr.predict(face_g)
        if confidence == -1:
            # print("cannot detect faces")
            continue
        text = "'%s': %.2f" % (name, confidence)
        cv2.putText(im, text, (x1 + dist, y2 - dist),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    # cv2.imshow("t", im)
    # cv2.waitKey(1)
    cv2.imwrite("./test/output/%s.png" % '.'.join(os.path.basename(imgP).split('.')[:-1]), im)
    # cv2.destroyWindow("t")
