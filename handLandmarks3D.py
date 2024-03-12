import time

runTimeRecorder = time.perf_counter()
print("loading dictionary")

import cv2
import mediapipe as mp
import matplotlib
import numpy
import copy
import matplotlib.pyplot as plt

import fps
import tools

MAX_FPS = 60
CAM_NUM = 0
MAX_HANDS_AMOUNT = 1

# 3d圖xyz比例記得調一樣
# 不然手會爛掉
#
#

time.perf_counter()
print("took " + str(time.perf_counter() - runTimeRecorder) + " sec")
runTimeRecorder = time.perf_counter()
print("loading camera")

cap = cv2.VideoCapture(CAM_NUM)

if cap is None or not cap.isOpened():
    raise Exception(
        'Unable to access camera, please check README.md for more info')

print("took " + str(time.perf_counter() - runTimeRecorder) + " sec")
runTimeRecorder = time.perf_counter()
print("loading model")

mpHands = mp.solutions.hands
handModel = mpHands.Hands(model_complexity=1,
                          max_num_hands=MAX_HANDS_AMOUNT,
                          min_detection_confidence=0.95,
                          min_tracking_confidence=0.05)
mpDraw = mp.solutions.drawing_utils

print("took " + str(time.perf_counter() - runTimeRecorder) + " sec")
runTimeRecorder = time.perf_counter()

fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(projection='3d')

# 初始化動態圖
plt.ion()

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
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

FPS = fps.fps()

while (True):

    curFps = FPS.get()

    ax.cla()
    ret, img = cap.read()

    if (ret):
        imgHigh = img.shape[0]
        imgWidth = img.shape[1]
        imgSize = (imgHigh + imgWidth) / 2

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        handResult = handModel.process(imgRGB)
        if (handResult.multi_hand_landmarks):
            handLandmark = numpy.zeros((21, 3))

            for i in range(21):
                handLandmark[i][0] = (handResult.multi_hand_landmarks[0].
                                      landmark[i].x) * imgWidth / imgSize
                handLandmark[i][1] = (handResult.multi_hand_landmarks[0].
                                      landmark[i].y) * imgHigh / imgSize
                handLandmark[i][2] = (handResult.multi_hand_landmarks[0].
                                      landmark[i].z) * imgWidth / imgSize

            handLandmarkDraw = numpy.zeros((3, 21))

            for i in range(21):
                handLandmarkDraw[0][i] = handLandmark[i][0]
                handLandmarkDraw[1][i] = 1 - handLandmark[i][1]
                handLandmarkDraw[2][i] = handLandmark[i][2]

            ax.plot(handLandmarkDraw[0], handLandmarkDraw[2],
                    handLandmarkDraw[1], 'ro')

            for a, b in hancLandmarkConnection:
                ax.plot([handLandmarkDraw[0][a], handLandmarkDraw[0][b]],
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

            ax.set_xlim(xRange)
            ax.set_ylim(zRange)
            ax.set_zlim(yRange)

            for handLms in handResult.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cv2.putText(img, "fps: " + str(int(curFps)), (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        cv2.imshow("cam", img)
    else:
        pass

    # fig.canvas.blit(fig.bbox)
    # 動態圖更新
    fig.canvas.flush_events()

    cv2KeyEvent = cv2.waitKey(1)
    if (cv2KeyEvent == ord('q')):
        break

    if (cv2KeyEvent == 27):
        break
