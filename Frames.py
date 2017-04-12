import time
from datetime import datetime as dt


class FrameLimiter:
    def __init__(self):
        self.lastFrameTime = None

    def limitFps(self, fps):
        if self.lastFrameTime is None:
            self.lastFrameTime = self.__now()
            return

        wait_sec = (1.0 / fps) - self.__elapsedSec()
        wait_sec = max(wait_sec, 0)

        time.sleep(wait_sec)
        self.lastFrameTime = self.__now()

    @staticmethod
    def __now():
        return dt.now()

    def __elapsedSec(self):
        return float((self.__now() - self.lastFrameTime).total_seconds())

class FpsCounter:
    def __init__(self):
        self.lastFrameTime = None

    @staticmethod
    def __now():
        return dt.now()

    def __elapsedSec(self):
        e = self.lastFrameTime
        if e is None:
            return 0.0
        return float((self.__now() - e).total_seconds())

    def actualFps(self):
        e = self.__elapsedSec()
        self.lastFrameTime = self.__now()
        if e == 0.0 or e is None:
            return 0.0
        return 1.0 / e
