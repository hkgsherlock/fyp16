from argparse import ArgumentParser
import time

import cv2
import imutils
import numpy as np
from imutils.video import FileVideoStream
from imutils.video import FPS

from FaceCascading import FaceCascadingOpencvHaar, FaceCascadingDlib
from BodyCascading import BodyCascading
from FaceRecognising import FaceRecognisingOpencv
from MotionDetection import NoWaitMotionDetection
from Performance.Frames import FrameLimiter
from Performance.Performance import TimeElapseCounter
from VideoRecorder import TaggingTimerVideoRecorder, NoWaitVideoRecorder


class CameraCapturing:
    def __init__(self, resolution=(854, 480), resolution_calc=(320, 180), framerate=30.0):
        ap = ArgumentParser()
        group = ap.add_mutually_exclusive_group()
        group.add_argument("-v", "--video", help="path to the video file")
        group.add_argument("-p", "--picam", help="use Raspberry Pi Camera", action='store_true')
        ap.add_argument("-r", "--rotate", help="rotate image by 180 deg", action='store_true')
        ap.add_argument("-w", "--showWnd", help="show window for the output image", action='store_true')
        args = vars(ap.parse_args())

        self.vid = None
        self.is_file = False
        self.show_wnd = args['showWnd']
        self.rotate180 = args.get("rotate", False)

        self.resolution = resolution
        self.resolution_calc = resolution_calc
        self.resolution_calc_multiply = resolution[0] / resolution_calc[0]
        self.framerate = framerate

        if args['picam'] is True:
            from imutils.video.pivideostream import PiVideoStream
            self.vid = PiVideoStream(self.resolution, self.framerate)
        elif args['video'] is not None:
            from imutils.video.filevideostream import FileVideoStream
            self.vid = FileVideoStream(args['video'], queueSize=256)
            self.is_file = True
        else:
            from imutils.video.webcamvideostream import WebcamVideoStream
            self.vid = WebcamVideoStream()

        self.vid.start()
        time.sleep(2)

        self.motion_detect = NoWaitMotionDetection()
        self.face_detect = FaceCascadingDlib()
        self.face_recognise = FaceRecognisingOpencv()
        self.face_recognise.prepare()
        self.video_write = NoWaitVideoRecorder()

    def main(self):
        try:
            lap = TimeElapseCounter()
            fl = FrameLimiter()
            while True:
                mat = self.vid.read()
                if mat is None:
                    time.sleep(1)
                    continue
                if self.rotate180:
                    mat = cv2.flip(mat, -1)
                grayMat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
                proc_mat = imutils.resize(grayMat, height=self.resolution_calc[-1])
                bb = np.multiply(self.motion_detect.putNewFrameAndCheck(proc_mat),
                                 float(self.resolution[-1]) / float(self.resolution_calc[-1]))
                bb = np.array(np.round(bb), 'int32')
                if len(bb) > 0:
                    x1, y1, x2, y2 = bb[0]
                    trim_for_face = grayMat[y1:y2, x1:x2]
                    faces = self.face_detect.detect_face_crop_frame(trim_for_face)
                    for face in faces:
                        who, conf = self.face_recognise.predict(face)
                    # TODO:
                    lap.start()
                if lap.is_started() and (0 < lap.lap() > 5):
                    self.video_write.endWrite()
                else:
                    self.video_write.write(mat)
                fl.limitFps(30)

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
        except KeyboardInterrupt, SystemExit:
            print("oops")

if __name__ == '__main__':
    camera = CameraCapturing()
    camera.main()
