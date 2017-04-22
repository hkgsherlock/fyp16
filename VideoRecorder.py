from Queue import Queue
from datetime import datetime
import time
from threading import Thread

import cv2

from DatabaseStorage import DatabaseStorage
from DropboxIntegration import DropboxIntegration
from GmailIntegration import GmailIntegration
from Performance.Frames import FrameLimiter


class Flag:
    def __init__(self, defVal=None):
        self.value = defVal


class NoWaitVideoRecorder:
    def __init__(self, width=854, height=480, fps=30):
        self.__fileName = None
        self.__vr = None
        self.__q = None
        self.__ending = None
        self.__t = None
        self.__fps = fps
        self.__pausing = Flag()
        self.width = width
        self.height = height
        self.fps = fps
        self.__people = []
        self.people_category = DatabaseStorage.get_faces_categories()
        self.last_frame = None

    def __ensureStarted(self, width=None, height=None, fps=None, fileName=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if fps is None:
            fps = self.fps

        if self.__t is None:
            self.__vr = VideoRecorder(width=width, height=height, fps=fps)
            self.__q = Queue()
            self.__ending = Flag()
            self.__fileName = Flag('')
            self.__people = []
            self.people_category = DatabaseStorage.get_faces_categories()
            self.__t = Thread(target=self.__update, args=[self.__vr, self.__q, self.__ending,
                                                          self.__pausing, self.__fileName])
            self.__t.daemon = True
            self.__t.start()

    def write(self, cv2mat, file_name=None):
        self.__ensureStarted()
        if file_name is not None:
            self.__fileName.value = file_name
        self.__q.put(cv2mat)
        self.last_frame = cv2mat

    def endWrite(self):
        t = self.__t
        self.__ending.value = True
        from DatabaseStorage import DatabaseStorage
        for p in self.__people:
            DatabaseStorage.set_record_face(self.__fileName.value, p)
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
    def __update(self, vr, q, endFlag, pauseFlag, fileNameFlag):
        from Performance.Frames import FpsCounter
        fps = FpsCounter()
        fl = FrameLimiter()
        lastFrame = None

        while not (endFlag.value and q.empty()):
            if not pauseFlag.value:
                if not q.empty():
                    lastFrame = q.get()
                if lastFrame is not None:
                    vr.write(lastFrame, fileNameFlag.value)
                # fl.limitFps(self.__fps)
                # print("writer fps: %.2f fps" % fps.actualFps())
        file_name = vr.creatingVideoFileName
        vr.endWrite()
        print('upload')
        DropboxIntegration.feed_video_file_path_for_upload_async(file_name)

    def tagPerson(self, name):
        if name not in self.__people:
            self.__people.append(name)
            if name == '_ignore':
                return
            elif name is '-1' or name == -1:
                GmailIntegration.notify_who_nowait(self.last_frame)
            else:
                if name not in self.people_category:
                    return
                category = self.people_category[name]
                category = category.lower()
                if category == 'confirmed':
                    GmailIntegration.notify_confirmed_nowait(name)
                elif category == 'deny':
                    GmailIntegration.notify_deny_nowait(self.last_frame)
                    from Alarming import Alarming
                    Alarming.buzz()

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
        self.creatingVideoFileName = ''

    def ensurePrepared(self, filename=None):
        if self.__outVideoWriter is None:
            if filename is None or len(filename) == 0:
                self.videoFileName = datetime.now().strftime("%Y%m%d-%H%M%S")
            else:
                self.videoFileName = filename
            self.__outVideoWriter = self.__createVideoWriter(filename=self.videoFileName,
                                                             width=self.width, height=self.height, fps=self.fps)
            self.creatingVideoFileName = "%s.avi" % self.videoFileName
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
