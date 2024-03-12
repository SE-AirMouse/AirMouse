import numpy
import matplotlib.pyplot as plt

import tools

# 3d圖xyz比例記得調一樣
# 不然手會爛掉
#
#
hancLandmarkConnection = numpy.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [11, 12],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [5, 9],
    [9, 13],
    [13, 17],
])


class handDrawer3D:

    def __init__(self) -> None:
        # 初始化動態圖
        plt.ion()

        self.fig = plt.figure(figsize=(6, 6))
        self.ax = plt.subplot(projection='3d')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.show()

    def draw(self, handLandmark: numpy.ndarray) -> None:
        self.ax.cla()
        handLandmarkDraw = numpy.zeros((3,21))
        for i in range(21):
            handLandmarkDraw[0][i] = handLandmark[i][0]
            handLandmarkDraw[1][i] = 0 - handLandmark[i][1]
            handLandmarkDraw[2][i] = handLandmark[i][2]

        self.ax.plot(handLandmarkDraw[0], handLandmarkDraw[2],
                     handLandmarkDraw[1], 'ro')

        for a, b in hancLandmarkConnection:
            self.ax.plot([handLandmarkDraw[0][a], handLandmarkDraw[0][b]],
                         [handLandmarkDraw[2][a], handLandmarkDraw[2][b]],
                         [handLandmarkDraw[1][a], handLandmarkDraw[1][b]])

        xyzCenter = handLandmarkDraw.sum(axis=1) / 21
        xCenter = xyzCenter[0]
        yCenter = xyzCenter[1]
        zCenter = xyzCenter[2]

        xyzRangeSize = tools.getLength(handLandmark[9] - handLandmark[17])
        xyzRangeSize = numpy.array([-xyzRangeSize, xyzRangeSize]) * 1.5
        xRange = numpy.array(xCenter + xyzRangeSize)
        yRange = numpy.array(yCenter + xyzRangeSize)
        zRange = numpy.array(zCenter + xyzRangeSize)

        self.ax.set_xlim(xRange)
        self.ax.set_ylim(zRange)
        self.ax.set_zlim(yRange)

        # fig.canvas.blit(fig.bbox)
        # 動態圖更新
        self.fig.canvas.flush_events()
