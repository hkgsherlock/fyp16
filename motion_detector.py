import argparse
import datetime
import imutils
import time
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
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

lastFrame = None
outVideoWriter = None
lastSeenOccupied = 0.0

while True:
    tFrameInit = time.time()
    (grabbed, frame) = camera.read()
    text = ""
    timestamp = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
    timestampFn = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    if not grabbed:
        break

    # frame = imutils.resize(frame, width=560)
    frame = cv2.resize(frame, (560, 315))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frameToLast = gray.copy()

    if lastFrame is None:
        lastFrame = gray
        continue

    frameDelta = cv2.absdiff(lastFrame, gray)

    # input, lower limit, upper limit,
    thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    sobelX = cv2.Sobel(thresh, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(thresh, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    # v-- python2
    # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # python 3
    (_, cnts, _) = cv2.findContours(sobelCombined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for those contours create bounding boxes (basic)
    boundingBoxes = []
    bbMargin = 50

    for c in cnts:
        if cv2.contourArea(c) < args["min_area"]:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        # tuning margin
        xw = x + w + bbMargin
        yh = y + h + bbMargin
        x -= bbMargin
        y -= bbMargin
        boundingBoxes.append([x, y, xw, yh])
        cv2.rectangle(frame, (x, y), (xw, yh), (0, 0, 255), 2)
        text = "Occupied"
        lastSeenOccupied = time.time()

    # conqeur overlapped bounding boxes with non-maxima suppression
    boundingBoxesNew = non_max_suppression(np.array(boundingBoxes), probs=None, overlapThresh=0.1)
    for (x1, y1, x2, y2) in boundingBoxesNew:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, '{} --> {}'.format(len(boundingBoxes), len(boundingBoxesNew)), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, timestamp,
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    if (time.time() - lastSeenOccupied < 10):  # hold for 10 secs
        # if outVideoWriter is None:
        #     outVideoWriter = cv2.VideoWriter("{}.avi".format(timestampFn), cv2.VideoWriter_fourcc(*'MJPG'), 30,
        #                                      (560, 315))
        # # print(frame)
        # outVideoWriter.write(frame)
        # TODO: move the video writing into another thread to reduce laggy feel of playing video
        lastSeenOccupied = time.time()
    else:
        if (outVideoWriter is not None) & (time.time() - lastSeenOccupied > 10.0):
            outVideoWriter.release()
            outVideoWriter = None

    lastFrame = frameToLast

    cv2.imshow("main", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Sobel X+Y", sobelCombined)
    # print("elapsed: {}".format(time.time() - tFrameInit))
    # print("1/fps = {}".format(1/fps))
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
