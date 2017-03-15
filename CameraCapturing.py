from argparse import ArgumentParser
import time

import cv2

from FaceCascading import FaceCascading
from PeopleCascading import PeopleCascading
from FaceRecognising import FaceRecognising
from MotionDetection import MotionDetection
from VideoRecorder import TaggingTimerVideoRecorder


class CameraCapturing:
    def __init__(self, resolution=(560, 315), resolution_calc=(256, 144), framerate=30.0):
        ap = ArgumentParser()
        group = ap.add_mutually_exclusive_group()
        group.add_argument("-v", "--video", help="path to the video file")
        group.add_argument("-p", "--picam", help="use Raspberry Pi Camera", action='store_true')
        ap.add_argument("-r", "--rotate", help="rotate image by 180 deg", action='store_true')
        ap.add_argument("-a", "--min-area", type=int, default=200, help="minimum area size")
        args = vars(ap.parse_args())

        self.videoSrc = args.get("video", None)
        self.usePiCam = args.get("picam", False)

        self.resolution = resolution
        self.resolution_calc = resolution_calc
        self.resolution_calc_multiply = resolution[0] / resolution_calc[0]
        self.framerate = framerate

        self.rotate180 = args.get("rotate", False)

        self.faceRecogniser = FaceRecognising()

        self.videoRecorder = TaggingTimerVideoRecorder()
        self.motionDetector = MotionDetection()
        self.peopleCascading = PeopleCascading()
        self.faceCascading = FaceCascading()

    def main(self):
        if self.videoSrc is None:
            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FPS, self.framerate)
            time.sleep(0.25)
            fps = 30.0
        elif self.usePiCam:
            from imutils.video.pivideostream import PiVideoStream
            picam = PiVideoStream(resolution=self.resolution, framerate=self.framerate)
            picam.start()
        else:
            cam = cv2.VideoCapture(self.videoSrc)
            fps = float(cam.get(cv2.CAP_PROP_FPS))

        print("FPS: {}".format(fps))

        outVideoWriter = None

        while True:
            tFrameInit = time.time()
            if self.usePiCam:
                # picam.update()
                frame = picam.read()
            else:
                _, frame = cam.read()  # last one is frame data

            if self.rotate180:
                frame = cv2.flip(frame, -1)

            frame_resize = cv2.resize(frame, self.resolution)

            # We copy another frame with smaller resolution to make processing faster (a bit).
            # Of course, the error rate should be risen
            frameGray = cv2.cvtColor(cv2.resize(frame_resize.copy(), self.resolution_calc), cv2.COLOR_BGR2GRAY)

            for (x1, y1, x2, y2) in self.motionDetector.putNewFrameAndCheck(frame):
                cv2.rectangle(frame_resize, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # when face detected:
            # * start recording
            # * continue to detect faces for people tagging
            # * start motion detection (till 2mins of no detected movement)

            foundPeople = []

            # for every people detected in the frame blob, find the (only) face and check who they are
            for (xA, yA, xB, yB) in self.peopleCascading.detect(frameGray):
                cv2.rectangle(frame_resize,
                              (int(xA * self.resolution_calc_multiply),
                               int(yA * self.resolution_calc_multiply)),
                              (int(xB * self.resolution_calc_multiply),
                               int(yB * self.resolution_calc_multiply)),
                              (255, 255, 0), 2)
                faces_pos = self.faceCascading.detect_face(frameGray)
                for xC, yC, xD, yD in faces_pos:
                    cv2.rectangle(frame_resize,
                                  (int(xC * self.resolution_calc_multiply),
                                   int(yC * self.resolution_calc_multiply)),
                                  (int(xD * self.resolution_calc_multiply),
                                   int(yD * self.resolution_calc_multiply)),
                                  (0, 255, 0), 2)
                for imgFace in self.faceCascading.detect_face_crop_frame(frameGray, faces_pos):
                    # what to do after you found a face?
                    label, confidence = self.faceRecogniser.predict(imgFace)
                    text = "detected face of {} w/ confidence={}".format(label, confidence)
                    # print(text)
                    cv2.putText(frame_resize, text, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # foundPeople.append(label)
                    # self.videoRecorder.setSeeing()

            # if the video recorder has been triggered to start,
            # start the motion detector to check movement
            # (not every frame has face)
            # if self.videoRecorder.isRecording():
            #     moves = self.motionDetector.putNewFrameAndCheck(frame)
            #     for (x1, y1, x2, y2) in moves:
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #         self.videoRecorder.setSeeing()

            # show the result of the detection and recognition
            cv2.imshow("main", frame_resize)

            # (BENNNNNNN) stop recording if time exceeds seconds active
            if self.videoRecorder.isRecording() and time.time() - self.videoRecorder.getLastSeen() > 30.0:
                self.videoRecorder.endWrite()
            # else, if found someone, or already started, start/continue recording
            elif len(foundPeople) > 0 or self.videoRecorder.isRecording():
                self.videoRecorder.write(frame_resize)

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
        if cam is not None:
            cam.release()
            cam = None
        if picam is not None:
            picam.stop()
            picam = None
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera = CameraCapturing()
    camera.main()
