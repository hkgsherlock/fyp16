import argparse
import time

import cv2
import imutils
import numpy as np

from FaceCascading import FaceCascadingOpencvHaar
from FaceRecognising import FaceRecognisingOpencv
from ImageCorrection import ImageCorrection
from MotionDetection import MotionDetection, NoWaitMotionDetection
from Performance.Performance import TimeElapseCounter
from Performance.Frames import FrameLimiter, FpsCounter
from VideoRecorder import NoWaitVideoRecorder


def filterImg(g):
    perf = TimeElapseCounter()
    perf.start()
    print("filter img")
    g = ImageCorrection.equalize(g)
    # g = ImageCorrection.claheCv2Mat(g)
    # g = ImageCorrection.sharpenKernelCv2Mat(g)
    g = ImageCorrection.sharpenGaussianCv2Mat(g)
    g = ImageCorrection.brightness(g, 25)
    g = ImageCorrection.contrast(g, 1.25)
    # g = ImageCorrection.normalizeCv2Mat(g)
    perf.printLap()
    return g

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
vr = NoWaitVideoRecorder(fps=10)
fl = FrameLimiter()
fps = FpsCounter()
lap = TimeElapseCounter()

face_detect = FaceCascadingOpencvHaar()
face_recognise = FaceRecognisingOpencv()

i = 0

try:
    while True:
        mat = vid.read()
        if mat is None:
            time.sleep(1)
            continue
        mat = cv2.flip(mat, -1)
        mat = imutils.resize(mat, height=360)
        bbMat = mat
        # bbMat = imutils.resize(mat, height=144)
        bbMat = cv2.cvtColor(bbMat, cv2.COLOR_BGR2GRAY)
        bb = md.putNewFrameAndCheck(bbMat)
        # bb = np.array(np.round(np.multiply(bb, 480./144.)), 'int32')
        for (x1, y1, x2, y2) in bb:
            cv2.rectangle(mat, (x1, y1), (x2, y2), (0, 0, 255), 2)
            w = x2 - x1
            h = y2 - y1
            bbxText = "w = %d h = %d pix = %d" % (w, h, w * h)
            print(bbxText)
            cv2.putText(mat, bbxText, (x1 + 20, y1 + 20), cv2.FONT_HERSHEY_DUPLEX, .6, (0, 0, 255), 1)

            y2 = y1 + (y2 - y1) / 3

            trim_for_face = bbMat[y1:y2, x1:x2]
            trim_for_face = filterImg(trim_for_face)
            # cv2.imwrite("test/processing/cut_frame/%d.jpg" % i, trim_for_face)
            i += 1
            perf_lap = TimeElapseCounter()
            perf_lap.start()
            faces_bb = face_detect.detect_face(trim_for_face)
            print("face detection used %.2f secs" % perf_lap.lap())
            for (xa, ya, xb, yb) in faces_bb:
                xa = xa + x1
                xb = xb + x1
                ya = ya + y1
                yb = yb + y1
                cv2.rectangle(mat, (xa, ya), (xb, yb), (0, 192, 0), 2)
            faces = face_detect.detect_face_crop_frame(trim_for_face, faces_bb)
            for f in faces:
                who, conf = face_recognise.predict(f)
                print("%s, %.2f" % (who, conf))

            lap.start()
        if lap.is_started() and (0 < lap.lap() > 5):
            vr.endWrite()
        else:
            vr.write(mat)
        fl.limitFps(10)
        actualFps = fps.actualFps()
        print("%.2f fps" % actualFps)
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
