import numpy
import fps
import mediapipe as mp
import cv2
import modelAccuracyTester

accRecorder = modelAccuracyTester.modelAccuracyRecorder("testing the code")

mpHands = mp.solutions.hands
mpHandModel = mpHands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.90,
    min_tracking_confidence=0.05,
)
mpDraw = mp.solutions.drawing_utils


FPS = fps.fps()
MAX_FPS = 60
cap = cv2.VideoCapture(0)


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


while True:
    curFps = FPS.get(limitFps=MAX_FPS)
    avgFps = FPS.avgFps()
    # print(round(avgFps))

    handLandmark = None

    ret, img = cap.read()

    if ret:
        # cv2.imshow("img1", img)
        imgHigh = img.shape[0]
        imgWidth = img.shape[1]
        imgSize = (imgHigh + imgWidth) / 2

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = mpHandModel.process(imgRGB)

        if result.multi_hand_landmarks:

            handLandmark = mpLandmarks2ndarray(
                landmarks=result.multi_hand_landmarks[0].landmark,
                correctScale=True,
                imgWidth=imgWidth,
                imgHigh=imgHigh,
            )

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
        cv2.imshow("cam", img)

    cv2KeyEvent = cv2.waitKey(1)

    if cv2KeyEvent == ord("s") and type(handLandmark) != type(None):
        accRecorder.setStandard(handLandmark)
    if cv2KeyEvent == ord("r") and type(handLandmark) != type(None):
        accRecorder.record(handLandmark)
    if cv2KeyEvent == ord("p") and type(handLandmark) != type(None):
        accRecorder.push(handLandmark)
    if cv2KeyEvent == ord("q"):
        break
    if cv2KeyEvent == 27:
        break

accRecorder.finish()
