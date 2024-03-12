import tools
import numpy


def gesturesName(fingerResult: numpy.ndarray) -> str:
    if ((fingerResult == numpy.array([0, 1, 0, 0, 0])).all()):
        return "leftClick"
    elif ((fingerResult == numpy.array([1, 1, 1, 1, 1])).all()):
        return "fist"
    elif ((fingerResult == numpy.array([1, 1, 0, 1, 1])).all()):
        return "fxxk you"
    else:
        return "others"


def analize(landmark: numpy.ndarray, returnDegree = False) -> numpy.ndarray:
    fingersDegree = numpy.zeros(5)

    fingerVectorDefination = numpy.array([[2, 3, 4], [5, 6, 8], [9, 10, 12],
                                          [13, 14, 16], [17, 18, 20]])

    fingersVector = numpy.zeros((5, 2, 3))
    for i in range(5):
        curDefination = fingerVectorDefination[i]
        points = numpy.zeros((3, 3))
        for j in range(3):
            curPoint = landmark[curDefination[j]]
            # points[j] = numpy.array([curPoint.x, curPoint.y])
            try:
                points[j][0] = curPoint.x
                points[j][1] = curPoint.y
                points[j][2] = curPoint.z
            except:
                points[j][0] = curPoint[0]
                points[j][1] = curPoint[1]
                points[j][2] = curPoint[2]

            # points[j][2] = 0

        fingersVector[i] = numpy.array(
            [points[0] - points[1], points[2] - points[1]])

    # print(fingersVector.shape)

    for i in range(5):
        fingersDegree[i] = tools.getDegree(fingersVector[i][0],
                                           fingersVector[i][1])
        

    fingersResult = numpy.zeros(5)
    fingersTriggerDegrees = numpy.array([150, 100, 100, 100, 100])
    for i in range(5):
        if (abs(fingersDegree[i]) < fingersTriggerDegrees[i]):
            fingersResult[i] = 1
        else:
            fingersResult[i] = 0

    for i in range(5):
        fingersDegree[i] = int(fingersDegree[i])
    # print(fingersDegree[0])

    if(returnDegree):
        return fingersDegree
    else:
        return fingersResult
