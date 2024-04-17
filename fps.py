import time
import numpy


class fps:

    def __init__(self, avgLen=30):
        self.lastTime = time.perf_counter()
        self.avgLen = avgLen
        self.avgFpsRecord = numpy.zeros(self.avgLen)

    def get(self, limitFps=999999):
        curTime = time.perf_counter()
        while (limitFps and curTime < self.lastTime + 1 / limitFps):
            curTime = time.perf_counter()
        try:
            ans = 1 / (curTime - self.lastTime)
        except:
            ans = float('inf')
        self.lastTime = curTime
        self.pushAvgFps(fps=ans)
        return ans

    def pushAvgFps(self, fps):
        for i in range(self.avgLen - 1, 0, -1):
            self.avgFpsRecord[i] = self.avgFpsRecord[i - 1]
        self.avgFpsRecord[0] = fps

    def avgFps(self):
        return self.avgFpsRecord.sum() / self.avgLen
