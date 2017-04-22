from argparse import ArgumentParser
import time

import cv2
import imutils
import numpy as np
from imutils.video import FileVideoStream
from imutils.video import FPS

from DatabaseStorage import DatabaseStorage
from FaceCascading import FaceCascadingOpencvHaar, FaceCascadingDlib
from FaceRecognising import FaceRecognisingOpencv
from GmailIntegration import GmailIntegration
from ImageCorrection import ImageCorrection
from MotionDetection import NoWaitMotionDetection
from Performance.Frames import FrameLimiter, FpsCounter
from Performance.Performance import TimeElapseCounter
from VideoRecorder import NoWaitVideoRecorder


class CameraCapturing:
    def __init__(self, resolution=(854, 480), resolution_calc=(320, 180), framerate=30.0,
                 videoSrc=None, picam=True, rotate180=True, showWnd=False,
                 motionDetect=None, faceDetect=None, faceRecognise=None, videoWrite=None):
        self.vid = None
        self.is_file = False
        self.show_wnd = showWnd
        self.rotate180 = rotate180

        self.resolution = resolution
        self.resolution_calc = resolution_calc
        self.resolution_calc_multiply = resolution[0] / resolution_calc[0]
        self.framerate = framerate

        if picam is True:
            from imutils.video.pivideostream import PiVideoStream
            self.vid = PiVideoStream(self.resolution, self.framerate)
        elif videoSrc is not None:
            from imutils.video.filevideostream import FileVideoStream
            self.vid = FileVideoStream(videoSrc, queueSize=256)
            self.is_file = True
        else:
            from imutils.video.webcamvideostream import WebcamVideoStream
            self.vid = WebcamVideoStream()

        self.vid.start()
        time.sleep(2)

        if motionDetect is not None:
            self.motion_detect = motionDetect
        else:
            self.motion_detect = NoWaitMotionDetection()

        if faceDetect is not None:
            self.face_detect = faceDetect
        else:
            self.face_detect = FaceCascadingOpencvHaar()

        if faceRecognise is not None:
            self.face_recognise = faceRecognise
        else:
            self.face_recognise = FaceRecognisingOpencv()

        if videoWrite is not None:
            self.video_write = videoWrite
        else:
            self.video_write = NoWaitVideoRecorder()

    @staticmethod
    def filterImg(g):
        g = ImageCorrection.equalize_cv2(g)
        g = ImageCorrection.sharpenGaussianCv2Mat(g)
        g = ImageCorrection.brightness(g, 25)
        g = ImageCorrection.contrast(g, 1.25)
        return g

    def main(self):
        fps = FpsCounter()
        try:
            lap = TimeElapseCounter()
            fl = FrameLimiter()
            from ServiceEntryPoint import ServiceEntryPoint
            while not ServiceEntryPoint.API_REQUEST_EXIT and not ServiceEntryPoint.API_REQUEST_REINIT:
                mat = self.vid.read()
                if mat is None:
                    time.sleep(1)
                    continue
                if self.rotate180:
                    mat = cv2.flip(mat, -1)
                bbMat = cv2.resize(mat, self.resolution_calc)
                # bbMat = imutils.resize(mat, height=144)
                bbMat = cv2.cvtColor(bbMat, cv2.COLOR_BGR2GRAY)
                bb = self.motion_detect.putNewFrameAndCheck(bbMat)
                # bb = np.array(np.round(np.multiply(bb, 480./144.)), 'int32')
                for (x1, y1, x2, y2) in bb:
                    y2 = y1 + (y2 - y1) / 3
                    trim_for_face = bbMat[y1:y2, x1:x2]
                    trim_for_face = self.filterImg(trim_for_face)
                    faces = self.face_detect.detect_face_crop_frame(trim_for_face)
                    for f in faces:
                        who, conf = self.face_recognise.predict(f)
                        self.video_write.tagPerson(who)
                        print("%s, %.2f" % (who, conf))
                    lap.start()
                if lap.is_started() and (0 < lap.lap() > 5):
                    self.video_write.endWrite()
                else:
                    self.video_write.write(mat)
                fl.limitFps(10)
                actualFps = fps.actualFps()
                print("%.2f fps" % actualFps)

                if self.show_wnd:
                    cv2.imshow("bbx", mat)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or (self.is_file and not self.vid.more()):
                        break
                    elif key == ord("p"):
                        while (cv2.waitKey(1) & 0xFF) != ord("p"):
                            if key == ord("q"):
                                break
                            continue
        except (KeyboardInterrupt, SystemExit):
            self.vid.stop()
            print("oops")
            raise SystemExit

if __name__ == '__main__':
    ap = ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("-v", "--video", help="path to the video file")
    group.add_argument("-p", "--picam", help="use Raspberry Pi Camera", action='store_true')
    ap.add_argument("-r", "--rotate", help="rotate image by 180 deg", action='store_true')
    ap.add_argument("-w", "--showWnd", help="show window for the output image", action='store_true')
    args = vars(ap.parse_args())

    camera = CameraCapturing(resolution=(854, 480), resolution_calc=(640, 360),
                             framerate=30.0, videoSrc=args['video'], picam=args['picam'],
                             rotate180=args.get("rotate", False), showWnd=args['showWnd'])
    camera.main()
