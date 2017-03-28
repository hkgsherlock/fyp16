import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image
from imutils.object_detection import non_max_suppression


def get_images_and_labels(path):
    images = []
    labels = []

    for dir in [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]:
        for image_fname in os.path.join(path, dir):
            image_path = os.path.join(path, dir, image_fname)
            image_pil = Image.open(image_path).convert('L')
            image = np.array(image_pil, 'uint8')
            nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            images.append(image)
            labels.append(nbr)
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
    for (xA, yA, xB, yB) in rects:
        imgFace = frame[yA:yB, xA:xB]
        # what to do after you found a face?


lastFrame = None
outVideoWriter = None
lastSeenOccupied = 0.0

# get the HOG Descriptor for image recognition
hog = cv2.HOGDescriptor()
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
recognizer = prepare_faceRecognizer()

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=200, help="minimum area size")
args = vars(ap.parse_args())

if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FPS, 30.0)
    time.sleep(0.25)
    fps = 30.0
else:
    camera = cv2.VideoCapture(args["video"])
    fps = float(camera.get(cv2.CAP_PROP_FPS))

# print("FPS: {}".format(fps))

while True:
    tFrameInit = time.time()
    (grabbed, frame) = camera.read()
    frame = cv2.resize(frame, (560, 315))
    frameGray = cv2.cvtColor(cv2.resize(frame.copy(), (256, 144)), cv2.COLOR_BGR2GRAY)

    # get the default people detector
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frameGray, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # for (x, y, w, h) in rects:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # for every people detected in the frame blob, find the (only) face and check who they are
    for (xA, yA, xB, yB) in rects:
        cv2.rectangle(frame,
                      (int(xA * 2.1875), int(yA * 2.1875)),
                      (int(xB * 2.1875), int(yB * 2.1875)),
                      (0, 255, 0), 2)
        detect_face(frame[yA:yB, xA:xB])

    # show the result of the detection and recognition
    cv2.imshow("main", frame)

    # codes to wait to stay the image as 30fps
    waitMs = int((1 / fps - (time.time() - tFrameInit)) * 1000)
    if waitMs < 1:
        waitMs = 1
    key = cv2.waitKey(waitMs) & 0xFF  # wait for key or if nothing then cont loop

    if key == ord("q"):
        break
    elif key == ord("p"):
        while (cv2.waitKey(1) & 0xFF) != ord("p"):
            continue

if outVideoWriter is not None:
    outVideoWriter.release()
camera.release()
cv2.destroyAllWindows()
