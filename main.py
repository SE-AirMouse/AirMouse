import time

runTimeRecorder = time.perf_counter()
print("loading dictionary")


import tools
import smoothHand
import mouseControl
import gesture
import fps
import copy
import numpy
import mediapipe as mp
import cv2


mouseControl = mouseControl.control()
handPosSmoother = smoothHand.smoothHand(smooth=30)
handSizeSmoother = smoothHand.smoothHand(smooth=30)
MAX_FPS = 60
CAM_NUM = 0
MAX_HANDS_AMOUNT = 10

mouseControlScale = int(2.5 * mouseControl.screenSize.sum() / 2)

time.perf_counter()
print("took " + str(time.perf_counter() - runTimeRecorder) + " sec")
runTimeRecorder = time.perf_counter()
print("loading camera")

cap = cv2.VideoCapture(CAM_NUM)

if cap is None or not cap.isOpened():
    raise Exception("Unable to access camera, please check README.md for more info")

print("took " + str(time.perf_counter() - runTimeRecorder) + " sec")
runTimeRecorder = time.perf_counter()
print("loading model")

mpHands = mp.solutions.hands
multiHandModel = mpHands.Hands(
    model_complexity=1,
    max_num_hands=MAX_HANDS_AMOUNT,
    min_detection_confidence=0.95,
    min_tracking_confidence=0.05,
)
singleHandModel = mpHands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.30,
    min_tracking_confidence=0.05,
)

mpDraw = mp.solutions.drawing_utils

print("took " + str(time.perf_counter() - runTimeRecorder) + " sec")
runTimeRecorder = time.perf_counter()

FPS = fps.fps()


# handControlActivationCount = 0
handControlActivationCount = numpy.zeros((MAX_HANDS_AMOUNT))
handControlState = "None"
handControlMatchingTarget = None


def mpLandmarks2ndarray(landmarks, correctScale=True, imgWidth=None, imgHigh=None):
    result = numpy.zeros((21, 3))

    if correctScale and not (imgHigh and imgWidth):
        raise Exception("input is not complete")
        return None

    try:
        for i in range(21):
            result[i][0] = landmarks[i].x
            result[i][1] = landmarks[i].y
            result[i][2] = landmarks[i].z

        if correctScale:
            imgSize = (imgWidth + imgHigh) / 2
            for i in range(21):
                result[i][0] = landmarks[i].x * imgWidth / imgSize
                result[i][1] = landmarks[i].y * imgHigh / imgSize
                result[i][2] = landmarks[i].z * imgWidth / imgSize

    except:
        pass

    return result


def handImageFilter3(img, landmarks: numpy.ndarray, mainLandmark):
    imgWidth = img.shape[1]
    imgHigh = img.shape[0]
    x1 = landmarks[mainLandmark].min(0)[0] * imgWidth
    y1 = landmarks[mainLandmark].min(0)[1] * imgHigh
    x2 = landmarks[mainLandmark].max(0)[0] * imgWidth
    y2 = landmarks[mainLandmark].max(0)[1] * imgHigh
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    x1 = (x1 - xc) * 1.5 + xc
    x2 = (x2 - xc) * 1.5 + xc
    y1 = (y1 - yc) * 1.5 + yc
    y2 = (y2 - yc) * 1.5 + yc
    x1 = max(0, int(x1))
    x2 = min(imgWidth, int(x2) + 1)
    y1 = max(0, int(y1))
    y2 = min(imgHigh, int(y2) + 1)
    tmpImg = copy.deepcopy(img)
    allLandmarksCount = len(landmarks)
    for i in range(allLandmarksCount):
        if i == mainLandmark:
            continue
        X1 = landmarks[i].min(0)[0] * imgWidth
        Y1 = landmarks[i].min(0)[1] * imgHigh
        X2 = landmarks[i].max(0)[0] * imgWidth
        Y2 = landmarks[i].max(0)[1] * imgHigh
        X1 = max(0, int(X1))
        X2 = min(imgWidth, int(X2) + 1)
        Y1 = max(0, int(Y1))
        Y2 = min(imgHigh, int(Y2) + 1)
        cv2.rectangle(img, (X1, Y1), (X2, Y2), (0, 0, 0), thickness=-1)

    img[y1:y2, x1:x2] = tmpImg[y1:y2, x1:x2]


class handSmoother:
    # translate landmarks into smooth hand status

    def __init__(self) -> None:
        self.landmarkSmoother = numpy.ndarray((21), dtype=object)
        for i in range(21):
            self.landmarkSmoother[i] = smoothHand.smoothHand(smooth=30)

    def set(self, handLandmarks: numpy.ndarray) -> None:
        # set by 1:1:1 scale 3d landmarks
        for i in range(21):
            self.landmarkSmoother[i].setPos(handLandmarks[i])

    def push(self, handLandmarks: numpy.ndarray) -> None:
        # push in 1:1:1 scale 3d landmarks
        for i in range(21):
            self.landmarkSmoother[i].pushPos(handLandmarks[i])

    def get(self) -> numpy.ndarray:
        # return smooth landmarks
        result = numpy.zeros((21, 3))
        for i in range(21):
            result[i] = self.landmarkSmoother[i].getPos()
        return result


class handControl:
    # control mouse by hand

    def __init__(self) -> None:
        self.lastHandPos = numpy.zeros((3))
        self.lastFingerTrigger = numpy.zeros((5))
        self.fistExitCount = 0
        self.controlling = False
        self.mouseSensitiveScale = 0.2
        self.fingersTriggerDegrees = numpy.array(
            [[140, 150], [100, 110], [100, 110], [100, 110], [110, 120]]
        )

        self.actionStatus = {
            "leftClickHold": False,
            "rightClickHold": False,
            "middleClickHold": False,
            "doubleClicked": False,
            "ctrlZooming": False,
            "adjustingMouseSensitive": False,
        }
        self.__try2actTranslate = {
            "left": "leftClickHold",
            "right": "rightClickHold",
            "middle": "middleClickHold",
            "double": "doubleClicked",
            "ctrl": "ctrlZooming",
        }

    def push(self, handLandmarks: numpy.ndarray) -> None:
        # push in 1:1:1 scale 3d landmarks

        self.__control(handLandmarks)

    def set(self, handLandmarks: numpy.ndarray) -> None:
        # set by 1:1:1 scale 3d landmarks
        self.lastHandPos = handLandmarks[13]
        self.__control(handLandmarks)

    def __control(self, handLandmarks: numpy.ndarray) -> None:
        deltaHandPos = handLandmarks[13] - self.lastHandPos
        self.lastHandPos = handLandmarks[13]
        handSize = tools.getLength(handLandmarks[5] - handLandmarks[17])
        deltaMousePos = (
            deltaHandPos * mouseControlScale * self.mouseSensitiveScale / handSize
        )
        deltaMousePos[0] *= -1
        self.curFingerstrigger = self.__fingerTriggerSwitch(handLandmarks)
        fingerTriggerToken = copy.deepcopy(self.curFingerstrigger)

        # slow mode
        if self.curFingerstrigger[4]:
            deltaMousePos *= 0.1
        else:
            pass

        # exit controll
        if (fingerTriggerToken >= numpy.array([1, 1, 1, 1, 1])).all():
            self.fistExitCount += 1
            if self.fistExitCount >= 5:
                self.mouseExit()
                self.controlling = False
            fingerTriggerToken -= numpy.array([1, 1, 1, 1, 1])
            return
        else:
            self.fistExitCount = 0

        # adjusting sensitive
        if (fingerTriggerToken >= numpy.array([0, 1, 1, 1, 0])).all():
            self.actionStatus["adjustingMouseSensitive"] = True
            if not self.curFingerstrigger[4]:
                self.mouseSensitiveScale += -(deltaHandPos[1] / handSize) * 0.1
            else:
                self.mouseSensitiveScale += -(deltaHandPos[1] / handSize) * 0.01

            showingSensitive = numpy.zeros((200, 600, 3), numpy.uint8)
            showingSensitive.fill(255)
            cv2.putText(
                showingSensitive,
                str(self.mouseSensitiveScale),
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                2,
            )
            cv2.imshow("sensitive", showingSensitive)
            cv2.setWindowProperty("sensitive", cv2.WND_PROP_TOPMOST, 1)
            fingerTriggerToken -= numpy.array([0, 1, 1, 1, 0])
        else:
            self.actionStatus["adjustingMouseSensitive"] = False
            try:
                cv2.destroyWindow("sensitive")
            except:
                pass
        # middle
        if (fingerTriggerToken >= numpy.array([0, 0, 1, 1, 0])).all():
            self.__try2act("middle")
            fingerTriggerToken -= numpy.array([0, 0, 1, 1, 0])
        else:
            self.__try2act("middle", hold=False)

        # double click
        if (fingerTriggerToken >= numpy.array([0, 1, 1, 0, 0])).all():
            self.__try2act("double")
            fingerTriggerToken -= numpy.array([0, 1, 1, 0, 0])
        else:
            self.__try2act("double", hold=False)

        # zooming
        if (fingerTriggerToken >= numpy.array([1, 0, 0, 1, 0])).all():
            self.__try2act("ctrl")
            mouseControl.scroll(deltaMousePos[1])
            fingerTriggerToken -= numpy.array([1, 0, 0, 1, 0])
        else:
            self.__try2act("ctrl", hold=False)

        # scrolling
        if (fingerTriggerToken >= numpy.array([0, 0, 0, 1, 0])).all():
            mouseControl.scroll(deltaMousePos[1] * 1.2)
            mouseControl.hscroll(deltaMousePos[0] * 1.2)
            fingerTriggerToken -= numpy.array([0, 0, 0, 1, 0])
        else:
            pass

        # left click
        if (fingerTriggerToken >= numpy.array([0, 1, 0, 0, 0])).all():
            self.__try2act("left")
            fingerTriggerToken -= numpy.array([0, 1, 0, 0, 0])
        else:
            self.__try2act("left", hold=False)

        # right click
        if (fingerTriggerToken >= numpy.array([0, 0, 1, 0, 0])).all():
            self.__try2act("right")
            fingerTriggerToken -= numpy.array([0, 0, 1, 0, 0])
        else:
            self.__try2act("right", hold=False)

        # moving
        if (fingerTriggerToken >= numpy.array([1, 0, 0, 0, 0])).all():
            mouseControl.addDis(deltaMousePos[0:2])
            fingerTriggerToken -= numpy.array([1, 0, 0, 0, 0])
        else:
            pass

    def mouseExit(self) -> None:
        self.__try2act(button="left", hold=False)
        self.__try2act(button="right", hold=False)
        self.__try2act(button="middle", hold=False)
        self.__try2act(button="ctrl", hold=False)
        self.__try2act(button="double", hold=False)
        if self.actionStatus["adjustingMouseSensitive"]:
            try:
                cv2.destroyWindow("sensitive")
            except:
                pass
            self.actionStatus["adjustingMouseSensitive"] = False

    def __try2act(self, button: str, hold=True) -> None:
        actionStatusName = self.__try2actTranslate[button]

        if hold and not self.actionStatus[actionStatusName]:
            if button == "left" or button == "right" or button == "middle":
                mouseControl.mouseDown(button=button)
            elif button == "double":
                mouseControl.mouseDoubleClick(button="left")
            else:
                mouseControl.keyDown(button=button)
            self.actionStatus[actionStatusName] = True

        elif not hold and self.actionStatus[actionStatusName]:
            if button == "left" or button == "right" or button == "middle":
                mouseControl.mouseUp(button=button)
            elif button == "double":
                pass
            else:
                mouseControl.keyUp(button=button)
            self.actionStatus[actionStatusName] = False

    def __fingerTriggerSwitch(self, handLandmarks: numpy.ndarray) -> numpy.ndarray:
        fingersTriggerDegrees = self.fingersTriggerDegrees
        fingerDegrees = gesture.analize(handLandmarks, returnDegree=True)
        result = copy.deepcopy(self.lastFingerTrigger)
        for i in range(5):
            if (
                not self.lastFingerTrigger[i]
                and fingerDegrees[i] <= fingersTriggerDegrees[i][0]
            ):
                result[i] = 1
            elif (
                self.lastFingerTrigger[i]
                and fingerDegrees[i] >= fingersTriggerDegrees[i][1]
            ):
                result[i] = 0

        self.lastFingerTrigger = copy.deepcopy(result)
        return result


controlHandSmoother = handSmoother()
controller = handControl()

# import draw3DHand
# handDrawer = draw3DHand.handDrawer3D()

while True:
    curFps = FPS.get(limitFps=MAX_FPS)
    avgFps = FPS.avgFps()
    # print(round(avgFps))

    ret, img = cap.read()

    if ret:
        # cv2.imshow("img1", img)
        imgHigh = img.shape[0]
        imgWidth = img.shape[1]
        imgSize = (imgHigh + imgWidth) / 2

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if handControlState != "Activated":
            # finding the hand to control

            # modelProcessTimeRecorder1 = time.perf_counter()
            result = multiHandModel.process(imgRGB)
            # modelProcessTimeRecorder2 = time.perf_counter()
            # modelProcessTime = modelProcessTimeRecorder2 - modelProcessTimeRecorder1
            # print("model process took",round(modelProcessTime*1000),"ms!")
            # print("=",round(1/modelProcessTime),"fps")

            try:
                allLandmarksCount = len(result.multi_hand_landmarks)
            except:
                allLandmarksCount = 0

            allLandmarks = numpy.zeros((allLandmarksCount, 21, 3))
            allLandmarksCorrected = numpy.zeros((allLandmarksCount, 21, 3))
            allGestures = numpy.zeros((allLandmarksCount, 5))

            for i in range(MAX_HANDS_AMOUNT):
                if i < allLandmarksCount:
                    allLandmarks[i] = mpLandmarks2ndarray(
                        result.multi_hand_landmarks[i].landmark, correctScale=False
                    )
                    allLandmarksCorrected[i] = mpLandmarks2ndarray(
                        result.multi_hand_landmarks[i].landmark,
                        correctScale=True,
                        imgWidth=imgWidth,
                        imgHigh=imgHigh,
                    )

                    allGestures[i] = gesture.analize(allLandmarksCorrected[i])
                    if (allGestures[i] == numpy.array([1, 0, 0, 0, 0])).all():
                        handControlActivationCount[i] += 1
                    else:
                        handControlActivationCount[i] -= 1

                    handControlActivationCount[i] = max(
                        0, min(10, handControlActivationCount[i])
                    )

                else:
                    handControlActivationCount[i] = 0

            if handControlActivationCount.max() < 5:
                handControlState = "None"
                handControlMatchingTarget = None
            else:
                for i in range(allLandmarksCount):
                    if handControlActivationCount[i] >= 5:
                        handControlState = "Matching"
                        handControlMatchingTarget = i
                        break

            if handControlState == "Matching":
                imgOneHand = copy.deepcopy(imgRGB)
                handImageFilter3(imgOneHand, allLandmarks, handControlMatchingTarget)
                resultOneHand = singleHandModel.process(imgOneHand)

                try:
                    for handLms in resultOneHand.multi_hand_landmarks:
                        mpDraw.draw_landmarks(
                            imgOneHand, handLms, mpHands.HAND_CONNECTIONS
                        )
                except:
                    pass

                cv2.imshow("Matching", imgOneHand)

                if resultOneHand.multi_hand_landmarks:
                    handControlState = "Activated"
                    controlHand = resultOneHand.multi_hand_landmarks[0].landmark
                    controlHand = mpLandmarks2ndarray(
                        controlHand,
                        correctScale=True,
                        imgWidth=imgWidth,
                        imgHigh=imgHigh,
                    )
                    controlHandSmoother.set(controlHand)
                    controller.controlling = True
                    controller.set(controlHand)
            try:
                for handLms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            except:
                pass
            cv2.putText(
                img,
                "fps: " + str(int(curFps)),
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                2,
            )
            cv2.imshow("Searching", img)

        else:
            # controlling

            resultOneHand = singleHandModel.process(imgRGB)

            if resultOneHand.multi_hand_landmarks:
                mainLandmark = resultOneHand.multi_hand_landmarks[0]

                mainLandmark = numpy.zeros((21, 3))
                mainLandmarkCorrected = numpy.zeros((21, 3))

                mainLandmark = mpLandmarks2ndarray(
                    resultOneHand.multi_hand_landmarks[0].landmark, correctScale=False
                )
                mainLandmarkCorrected = mpLandmarks2ndarray(
                    resultOneHand.multi_hand_landmarks[0].landmark,
                    correctScale=True,
                    imgWidth=imgWidth,
                    imgHigh=imgHigh,
                )
                controlHandSmoother.push(mainLandmarkCorrected)
                smoothControlHand = controlHandSmoother.get()

                # handDrawer.draw(smoothControlHand)

                controller.push(smoothControlHand)
                if controller.controlling == False:
                    controller.mouseExit()
                    handControlState = "None"
                    # incase the model mis-track the wrong hand later which has the similiar position with the just-actively-quited one
                    while True:
                        resultQuitControlling = singleHandModel.process(
                            numpy.zeros((1, 1, 3), numpy.uint8)
                        )
                        if not resultQuitControlling.multi_hand_landmarks:
                            break

                for handLms in resultOneHand.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                cv2.putText(
                    img,
                    "fps: " + str(int(curFps)),
                    (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    2,
                )

                cv2.imshow("Controlling", img)

            else:
                handControlState = "None"
                controller.mouseExit()
                handControlActivationCount.fill(0)

    if handControlState != "Matching":
        try:
            cv2.destroyWindow("Matching")
            pass
        except:
            pass
    if handControlState != "Activated":
        try:
            cv2.destroyWindow("Controlling")
        except:
            pass
    else:
        try:
            cv2.destroyWindow("Searching")
        except:
            pass

    cv2KeyEvent = cv2.waitKey(1)
    if cv2KeyEvent == ord("q"):
        break

    if cv2KeyEvent == 27:
        break

cap.release()
