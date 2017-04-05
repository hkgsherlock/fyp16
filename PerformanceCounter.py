from datetime import datetime as dt


class TimeElapseCounter:
    def __init__(self):
        self.tStart = None
        self.laps = []

    def start(self):
        self.tStart = dt.now()

    def printStart(self):
        self.start()
        print('perf: start lap')

    def lap(self):
        if self.tStart is not None:
            sec = (self.now() - self.tStart).total_seconds()
        else:
            self.start()
            sec = 0
        self.laps.append(sec)
        return sec

    @staticmethod
    def now():
        return dt.now()

    def clearLaps(self):
        del self.laps[:]

    def printLap(self):
        sec = self.lap()
        print('perf: cost %d secs' % sec)

    def printLaps(self):
        print('perf: laps:\n%s' % ('\n'.join(['%s secs' % a for a in self.laps])))
