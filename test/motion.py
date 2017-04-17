import argparse
import time

import cv2
import imutils
import numpy as np

from MotionDetection import MotionDetection, NoWaitMotionDetection
from Performance.Performance import TimeElapseCounter
from Performance.Frames import FrameLimiter, FpsCounter
from VideoRecorder import NoWaitVideoRecorder

vid = None
file = False

ap = argparse.ArgumentParser()
gp = ap.add_mutually_exclusive_group()
gp.add_argument("-v", "--video", help="path to the video file")
gp.add_argument("-p", "--picam", help="use Raspberry Pi Camera", action='store_true')
ap.add_argument("-w", "--showWnd", help="show window for the output image", action='store_true')
ap.add_argument("-a", "--min-area", type=int, default=200, help="minimum area size")
args = vars(ap.parse_args())

if args['picam'] is True:
    from imutils.video.pivideostream import PiVideoStream

    vid = PiVideoStream((854, 480), 30)
elif args['video'] is not None:
    from imutils.video.filevideostream import FileVideoStream

    vid = FileVideoStream(args['video'], queueSize=256)
    file = True
else:
    from imutils.video.webcamvideostream import WebcamVideoStream

    vid = WebcamVideoStream()
vid.start()
time.sleep(2)

lastFrame = None
outVideoWriter = None

md = NoWaitMotionDetection()
vr = NoWaitVideoRecorder()
fl = FrameLimiter()
fps = FpsCounter()
lap = TimeElapseCounter()

try:
    while True:
        mat = vid.read()
        if mat is None:
            time.sleep(1)
            continue
        mat = cv2.flip(mat, -1)
        mat = imutils.resize(mat, height=480)
        bbMat = imutils.resize(mat, height=180)
        bb = np.multiply(md.putNewFrameAndCheck(bbMat), 480./180.)
        bb = np.round(bb)
        bb = np.array(bb, 'int32')
        for (x1, y1, x2, y2) in bb:
            cv2.rectangle(mat, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if len(bb) == 1:
                w = x2 - x1
                h = y2 - y1
                bbxText = "w = %d h = %d pix = %d" % (w, h, w * h)
                print(bbxText)
                cv2.putText(mat, bbxText, (x1 + 20, y1 + 20), cv2.FONT_HERSHEY_DUPLEX, .6, (0, 0, 255), 1)
                lap.start()
        if lap.is_started() and (0 < lap.lap() > 5):
            vr.endWrite()
        else:
            vr.write(mat)
        fl.limitFps(30)
        actualFps = fps.actualFps()
        # print("%.2f fps" % actualFps)
        cv2.putText(mat, "fps = %.2f" % actualFps, (30, 30), cv2.FONT_HERSHEY_DUPLEX, .6, (0, 192, 0), 1)
        if args['showWnd']:
            cv2.imshow("bbx", mat)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or (file and not vid.more()):
                break
            elif key == ord("p"):
                while (cv2.waitKey(1) & 0xFF) != ord("p"):
                    if key == ord("q"):
                        break
                    continue
except KeyboardInterrupt, SystemExit:
    print("oops")

print("waiting")
vr.endWriteWaitJoin()
print("ok")
vid.stop()
cv2.destroyAllWindows()
