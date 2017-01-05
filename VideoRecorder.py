from datetime import datetime, time

import cv2


def _createVideoWriter(filename, width=560, height=315, fps=30):
    return cv2.VideoWriter("{}.avi".format(filename),
                           cv2.VideoWriter_fourcc(*'MJPG'),
                           fps,
                           (width, height))


class VideoRecorder:
    def __init__(self):
        self.outVideoWriter = None
        self.videoFileName = ""

    def write(self, frame):
        if self.outVideoWriter is None:
            self.videoFileName = datetime.now().strftime("%d%m%Y-%H%M%S")
            self.outVideoWriter = _createVideoWriter(filename=self.videoFileName,
                                                     width=560, height=315, fps=30)
        if not self.outVideoWriter.isOpened():
            print("video writer cannot be opened")
            raise
        self.outVideoWriter.write(frame)

    def endWrite(self):
        if self.outVideoWriter is not None:
            self.outVideoWriter.release()
            self.outVideoWriter = None

    def isRecording(self):
        return self.outVideoWriter is not None and self.outVideoWriter.isOpened()


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
