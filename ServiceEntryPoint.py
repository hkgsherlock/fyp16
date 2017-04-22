from multiprocessing import Process

from CameraCapturing import CameraCapturing
from DatabaseStorage import DatabaseStorage
from FaceCascading import *
from FaceRecognising import FaceRecognisingOpencv
from MotionDetection import NoWaitMotionDetection
from StreamingAndWebApi import StreamingAndWebApi, ServerShutdown
from VideoRecorder import NoWaitVideoRecorder

import sys, signal


class ServiceEntryPoint:
    API_REQUEST_REINIT = False
    API_REQUEST_EXIT = False

    def __init__(self):
        self.__imgproc = None
        self.__api = None
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal, frame):
        self.API_REQUEST_EXIT = True
        self.__imgproc.terminate()
        self.__api.terminate()
        print("\nprogram exiting gracefully")
        sys.exit(0)

    def run(self):
        while not self.API_REQUEST_EXIT:
            if self.__imgproc is None:
                self.__imgproc = Process(target=self.run_imgproc)
                self.__imgproc.start()
            if self.__api is None:
                self.__api = Process(target=self.run_api)
                self.__api.start()
            if self.API_REQUEST_REINIT or self.API_REQUEST_EXIT:
                self.__imgproc.terminate()
                self.__api.terminate()
                self.API_REQUEST_REINIT = False
        print('shutdown service')

    def run_imgproc(self):
        try:
            cap = self.create_camera_capture_from_set(
                motion=self.create_motion_detect_from_set(),
                face=self.create_face_detect_from_set(),
                facerec=self.create_face_recognise_from_set(),
                videowrite=self.create_video_write_from_set()
            )
            cap.main()
        except (KeyboardInterrupt, SystemExit):
            pass

    def run_api(self):
        try:
            StreamingAndWebApi()
        except (ServerShutdown, SystemExit):
            pass

    def create_camera_capture_from_set(self, motion, face, facerec, videowrite):
        param = DatabaseStorage.get_capture_params()
        capture_width = int(param['capture']['width'])
        capture_height = int(param['capture']['height'])
        capture_frame_rate = float(param['capture']['frame_rate'])
        process_width = int(param['process']['width'])
        process_height = int(param['process']['height'])
        resolution = (capture_width, capture_height)
        resolution_calc = (process_width, process_height)
        return CameraCapturing(resolution=resolution,
                               resolution_calc=resolution_calc,
                               framerate=capture_frame_rate,
                               picam=True,
                               motionDetect=motion,
                               faceDetect=face,
                               faceRecognise=facerec,
                               videoWrite=videowrite)

    def create_motion_detect_from_set(self):
        param = DatabaseStorage.get_motion_params()
        threshold_low = int(param['threshold_low'])
        minimum_area = int(param['minimum_area'])
        bounding_box_padding = int(param['bounding_box_padding'])
        frame_span = int(param['frame_span'])
        return NoWaitMotionDetection(thresholdLow=threshold_low,
                                     minAreaSize=minimum_area,
                                     boundingBoxPadding=bounding_box_padding,
                                     frameSpan=frame_span)

    def create_face_detect_from_set(self):
        param = DatabaseStorage.get_face_params()
        str_face_method = param['face_method']
        if str_face_method == 'FaceCascadingOpencvHaar':
            return FaceCascadingOpencvHaar()
        if str_face_method == 'FaceCascadingOpencvLbp':
            return FaceCascadingOpencvLbp()
        if str_face_method == 'FaceCascadingOpencvDlib':
            return FaceCascadingDlib()

    def create_face_recognise_from_set(self):
        param = DatabaseStorage.get_facerec_params()
        str_facerec_method = param['facerec_method']
        method = None
        if str_facerec_method == 'createEigenFaceRecognizer':
            method = cv2.face.createEigenFaceRecognizer
        if str_facerec_method == 'createFisherFaceRecognizer':
            method = cv2.face.createFisherFaceRecognizer
        if str_facerec_method == 'createLBPHFaceRecognizer':
            method = cv2.face.createLBPHFaceRecognizer
        return FaceRecognisingOpencv(createMethod=method)

    def create_video_write_from_set(self):
        param = DatabaseStorage.get_record_params()
        record_width = int(param['record_width'])
        record_height = int(param['record_height'])
        record_framerate = float(param['record_framerate'])
        return NoWaitVideoRecorder(width=record_width, height=record_height, fps=record_framerate)

if __name__ == '__main__':
    ServiceEntryPoint().run()
