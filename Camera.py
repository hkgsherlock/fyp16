import argparse

import cv2
import time

import datetime

import FaceCascading
import PeopleCascading


class Camera:
    def __init__(self):
        pass

    def run(self):
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

        lastFrame = None
        outVideoWriter = None
        lastSeenOccupied = 0.0

        while True:
            tFrameInit = time.time()
            (grabbed, frame) = camera.read()
            frame = cv2.resize(frame, (560, 315))

            # We copy another frame with smaller resolution to make processing faster (a bit).
            # Of course, the error rate should be risen
            frameGray = cv2.cvtColor(cv2.resize(frame.copy(), (256, 144)), cv2.COLOR_BGR2GRAY)

            # for every people detected in the frame blob, find the (only) face and check who they are
            for (xA, yA, xB, yB) in PeopleCascading.detect(frameGray):
                cv2.rectangle(frame,
                              (int(xA * 2.1875), int(yA * 2.1875)),
                              (int(xB * 2.1875), int(yB * 2.1875)),
                              (0, 255, 0), 2)
                FaceCascading.detect_face(frameGray)

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


if __name__ == '__main__':
    Camera().run()
