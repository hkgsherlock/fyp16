import argparse
import datetime
import imutils
import time
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

lastFrame = None
outVideoWriter = None
lastSeenOccupied = 0.0

def detect_face(frame):
    rects, weights = face_cascade.detectMultiScale(frame, 1.3, 5)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    for (xA, yA, xB, yB) in rects:
        imgFace = frame[xA:yA, xB:yB]
        # TODO: what to do after you found a face?

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

print("FPS: {}".format(fps))

# get the HOG Descriptor for image recognition
hog = cv2.HOGDescriptor()

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

    for (xA, yA, xB, yB) in rects:
        cv2.rectangle(frame,
                      (int(xA * 2.1875), int(yA * 2.1875)),
                      (int(xB * 2.1875), int(yB * 2.1875)),
                      (0, 255, 0), 2)
        detect_face(frame[xA:yA, xB:yB])

    cv2.imshow("main", frame)

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
