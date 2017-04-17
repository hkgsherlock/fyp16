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
        if self.is_started():
            sec = float((self.now() - self.tStart).total_seconds())
        else:
            self.start()
            sec = 0.0
        self.laps.append(sec)
        return sec

    def is_started(self):
        return self.tStart is not None

    @staticmethod
    def now():
        return dt.now()

    def clearLaps(self):
        del self.laps[:]

    def printLap(self):
        sec = self.lap()
        print('perf: cost %.4f secs' % sec)

    def printLaps(self):
        print('perf: laps:\n%s' % ('\n'.join(['%4.f secs' % a for a in self.laps])))
