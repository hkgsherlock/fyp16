from Queue import Queue
from datetime import datetime
import time
from threading import Thread

import cv2

from Performance.Frames import FrameLimiter


class Flag:
    def __init__(self, defVal=None):
        self.value = defVal


class NoWaitVideoRecorder:
    def __init__(self, width=854, height=480, fps=30):
        self.fileName = None
        self.__vr = None
        self.__q = None
        self.__ending = None
        self.__t = None
        self.__fps = fps
        self.__pausing = Flag()
        self.width = width
        self.height = height
        self.fps = fps

    def __ensureStarted(self, width=None, height=None, fps=None, fileName=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if fps is None:
            fps = self.fps
        self.fileName = fileName

        if self.__t is None:
            self.__vr = VideoRecorder(width=width, height=height, fps=fps)
            self.__q = Queue()
            self.__ending = Flag()
            self.__t = Thread(target=self.__update, args=[self.__vr, self.__q, self.__ending, self.__pausing])
            self.__t.daemon = True
            self.__t.start()

    def write(self, cv2mat, fileName=None):
        self.fileName = fileName
        self.__ensureStarted()
        self.__q.put(cv2mat)

    def endWrite(self):
        t = self.__t
        self.__ending.value = True
        self.__t = None # make it headless
        return t

    def getWritePausing(self):
        return self.__pausing.value

    def setWritePausing(self, value):
        self.__pausing.value = value

    def endWriteWaitJoin(self, timeout=None):
        t = self.endWrite()
        if t is not None:
            t.join(timeout=timeout)

    # def __update(self, vr=VideoRecorder(), q=Queue(), endFlag=Flag()):
    def __update(self, vr, q, endFlag, pauseFlag):
        from Performance.Frames import FpsCounter
        fps = FpsCounter()
        fl = FrameLimiter()
        lastFrame = None

        while not (endFlag.value and q.empty()):
            if not pauseFlag.value:
                if not q.empty():
                    lastFrame = q.get()
                if lastFrame is not None:
                    vr.write(lastFrame, self.fileName)
                # fl.limitFps(self.__fps)
                # print("writer fps: %.2f fps" % fps.actualFps())
        print("ending")
        vr.endWrite()


class VideoRecorder:
    @staticmethod
    def __createVideoWriter(filename, width=640, height=360, fps=30):
        return cv2.VideoWriter("%s.avi" % filename,
                               cv2.VideoWriter_fourcc(*'MJPG'),
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
