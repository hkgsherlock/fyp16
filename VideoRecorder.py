from Queue import Queue
from datetime import datetime
import time
from threading import Thread

import cv2


class Flag:
    def __init__(self, defVal=False):
        self.value = defVal


class NoWaitVideoRecorder:
    def __init__(self, width=854, height=480, fps=30):
        self.fileName = None
        self.__vr = None
        self.__q = None
        self.__ending = None
        self.__t = None
        self.__ensureStarted(width=width, height=height, fps=fps)

    def __ensureStarted(self, width=854, height=480, fps=30, fileName=None):
        if self.__t is None:
            self.__vr = VideoRecorder(width=width, height=height, fps=fps)
            self.__q = Queue()
            self.__ending = Flag()
            self.__t = Thread(target=self.__update, args=[self.__vr, self.__q, self.__ending])
            self.__t.daemon = True
            self.__t.start()

    def write(self, cv2mat, fileName=None):
        self.fileName = fileName
        self.__ensureStarted()
        self.__q.put(cv2mat)

    def endWrite(self):
        self.__ending.value = True
        self.__t = None # make it headless

    # def __update(self, vr=VideoRecorder(), q=Queue(), endFlag=Flag()):
    def __update(self, vr, q, endFlag):
        while not endFlag.value:
            if not q.empty():
                vr.write(q.get(), )
        vr.endWrite()


class VideoRecorder:
    @staticmethod
    def __createVideoWriter(filename, width=640, height=360, fps=30):
        return cv2.VideoWriter("%s.avi" % filename,
                               cv2.VideoWriter_fourcc(*'H264'),
                               fps,
                               (width, height))

    def __init__(self, width=854, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.__outVideoWriter = None
        self.videoFileName = ""

    def ensurePrepared(self, filename=None):
        if self.__outVideoWriter is None:
            if filename is None:
                self.videoFileName = datetime.now().strftime("%Y%m%d-%H%M%S")
            else:
                self.videoFileName = filename
            self.__outVideoWriter = self.__createVideoWriter(filename=self.videoFileName,
                                                             width=self.width, height=self.height, fps=self.fps)
            time.sleep(1)
        # if not self.__outVideoWriter.isOpened():
        #     print("video writer cannot be opened")
        #     raise

    def write(self, frame, filename=None):
        self.ensurePrepared(filename)
        frame = cv2.resize(frame, (self.width, self.height))
        self.__outVideoWriter.write(frame)

    def endWrite(self):
        if self.__outVideoWriter is not None:
            self.__outVideoWriter.release()
            self.__outVideoWriter = None

    def isRecording(self):
        return self.__outVideoWriter is not None and self.__outVideoWriter.isOpened()


class TaggingTimerVideoRecorder(object, VideoRecorder):
    def __init__(self):
        VideoRecorder.__init__(self)
        self.people = []
        self.lastSeen = 0

    def setSeeing(self):
        self.lastSeen = time.time()

    def getLastSeen(self):
        return self.lastSeen

    def setPerson(self, label):
        if label in self.people:
            pass
        self.people.append(label)
        with open("{}.txt".format(self.videoFileName), "a") as tagLog:
            tagLog.write("{}\r\n".format(label))

    def endWrite(self):
        super(TaggingTimerVideoRecorder, self).endWrite()
        self.people = []
