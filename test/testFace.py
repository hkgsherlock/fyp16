import os

import cv2
import imutils

from FaceCascading import FaceCascadingOpencvHaar, FaceCascadingOpencvLbp
from FaceRecognising import FaceRecognisingOpencv
from Performance.Performance import TimeElapseCounter
from ImageCorrection import ImageCorrection

import argparse


def run():
    ap = argparse.ArgumentParser()
    gp = ap.add_mutually_exclusive_group()
    gp.add_argument("-b", "--batch", metavar='BATCH_PATH',
                    help='path of the batch job folder to be processed\n'
                         'for example:"./face/{name}/ok_{code}.*"')
    ap.add_argument("-r", "--resize", type=int, default=240, help="resize processing image to")
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

    resize = float(args['resize'])

    fc = FaceCascadingOpencvLbp()
    fr = FaceRecognisingOpencv()
    perf = TimeElapseCounter()

    for imgP in files:
        print(imgP)
        im = cv2.imread(imgP)
        height, width = im.shape[:-1]
        if height > resize and width > resize:
            if height > width:
                height = int(resize / width * height)
                width = resize
            else:
                width = int(resize / height * width)
                height = resize
            im = cv2.resize(im, (int(width), int(height)))
        g = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY)
        # go = g.copy()

        g = filterImg(g)

        # g2 = g.copy()

        # merge_debvec = cv2.createMergeDebevec()
        # hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())
        # tonemap1 = cv2.createTonemapDurand(gamma=2.2)
        # res_debvec = tonemap1.process(hdr_debvec.copy())

        # cv2.imshow("eq", g)
        # cv2.waitKey()
        # cv2.destroyWindow("eq")
        cv2.imwrite("./test/processing/%s.png" % '.'.join(os.path.basename(imgP).split('.')[:-1]), g)
        perf.start()
        print("detect")
        detected_faces = fc.detect_face(g)
        perf.printLap()
        # print("detected faces: %d" % len(detected_faces))
        i = 0
        for x1, y1, x2, y2 in detected_faces:
            # print("%d,%d,%d,%d" % (x1, y1, x2, y2))
            dist = int((x2 - x1) * .05)
            face_g = g[y1:y2, x1:x2]
            # face_g = filterImg(face_g)

            # for ex1, ey1, ex2, ey2 in eyeCc.detectMultiScale(
            #     face_g,
            #     scaleFactor=1.1,
            #     minNeighbors=5,
            #     flags=cv2.CASCADE_SCALE_IMAGE
            # ):
            #     # print("%d,%d,%d,%d" % (x1, y1, x2, y2))
            #     cv2.rectangle(im, (ex1 + x1, ey1 + x1), (ex2 + x1, ey2 + x1), (0, 255, 0), 5)

            perf.start()
            print("predict")
            name, confidence = fr.predict(face_g)
            perf.printLap()

            if name == '_ignore':
                continue
            elif name == -1:
                confidence = -1.0
            print("%s, conf = %.2f" % (name, confidence))
            cv2.imwrite("./test/output/_extract_face/%d_%s_%d__%s.jpg" % (i,
                                                                          name,
                                                                          int(confidence),
                                                                          '.'.join(
                                                                              os.path.basename(imgP).split('.')[:-1])),
                        cv2.resize(face_g, (200, 200)))
            i += 1
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 5)
            text = "'%s': %.2f" % (name, confidence)
            cv2.putText(im, text, (x1 + dist, y2 - dist),
                        cv2.FONT_HERSHEY_DUPLEX, .6, (0, 0, 255), 1)
        # cv2.imshow("t", im)
        # cv2.waitKey(1)
        cv2.imwrite("./test/output/%s.png" % '.'.join(os.path.basename(imgP).split('.')[:-1]), im)
        # cv2.destroyWindow("t")


def filterImg(g):
    perf = TimeElapseCounter()
    perf.start()
    print("filter img")
    g = ImageCorrection.equalize(g)
    g = ImageCorrection.claheCv2Mat(g)
    # g = ImageCorrection.sharpenKernelCv2Mat(g)
    g = ImageCorrection.sharpenGaussianCv2Mat(g)
    g = ImageCorrection.brightness(g, 25)
    g = ImageCorrection.contrast(g, 1.25)
    # g = ImageCorrection.normalizeCv2Mat(g)
    perf.printLap()
    return g

if __name__ == '__main__':
    run()
