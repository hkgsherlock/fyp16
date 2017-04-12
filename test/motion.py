import argparse
import time

import cv2

from Frames import FrameLimiter, FpsCounter
from MotionDetection import MotionDetection
import imutils
from imutils.video.fps import FPS

from VideoRecorder import VideoRecorder, NoWaitVideoRecorder

vid = None
file = False

ap = argparse.ArgumentParser()
gp = ap.add_mutually_exclusive_group()
gp.add_argument("-v", "--video", help="path to the video file")
gp.add_argument("-p", "--picam", help="use Raspberry Pi Camera", action='store_true')
ap.add_argument("-a", "--min-area", type=int, default=200, help="minimum area size")
args = vars(ap.parse_args())

if args['picam'] is True:
    from imutils.video.pivideostream import PiVideoStream

    vid = PiVideoStream((960, 720), 30)
elif args['video'] is not None:
    from imutils.video.filevideostream import FileVideoStream

    vid = FileVideoStream(args['video'], queueSize=256)
    file = True
else:
    from imutils.video.webcamvideostream import WebcamVideoStream

    vid = WebcamVideoStream()
time.sleep(2)
vid.start()

lastFrame = None
outVideoWriter = None

md = MotionDetection()
vr = NoWaitVideoRecorder()
fl = FrameLimiter()
fps = FpsCounter()

while True:
    mat = vid.read()
    mat = imutils.resize(mat, height=480)
    bb = md.putNewFrameAndCheck(mat)
    for (x1, y1, x2, y2) in bb:
        cv2.rectangle(mat, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if len(bb) == 1:
            w = x2 - x1
            h = y2 - y1
            cv2.putText(mat, "w = %d h = %d pix = %d" % (w, h, w * h), (x1 + 20, y1 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, .6, (0, 0, 255), 1)
    fl.limitFps(30)
    cv2.putText(mat, "fps = %.2f" % fps.actualFps(), (30, 30),
                cv2.FONT_HERSHEY_DUPLEX, .6, (0, 192, 0), 1)
    cv2.imshow("bbx", mat)
    vr.write(mat, fileName="motion_test")
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or (file and not vid.more()):
        break
    elif key == ord("p"):
        while (cv2.waitKey(1) & 0xFF) != ord("p"):
            if key == ord("q"):
                break
            continue
vid.stop()
vr.endWrite()
cv2.destroyAllWindows()
