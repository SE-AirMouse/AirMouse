import math
import numpy


def getVectorLength(vector: numpy.ndarray) -> float:
    return ((vector**2).sum())**0.5


def getLength(vector: numpy.ndarray) -> float:
    return ((vector**2).sum())**0.5


def getDegree(vector_a: numpy.ndarray, vector_b: numpy.ndarray) -> float:
    cosTheta = (vector_a * vector_b).sum() / (getVectorLength(vector_a) *
                                              getVectorLength(vector_b))
    if (cosTheta > 1):
        cosTheta = 1
    elif (cosTheta < -1):
        cosTheta = -1
    theda = math.acos(cosTheta)
    return math.degrees(theda)

def externalProduct(vector_a: numpy.ndarray, vector_b: numpy.ndarray) -> numpy.ndarray:
    ans = numpy.zeros((3))
    ans[0] = vector_a[1]*vector_b[2] - vector_b[1]*vector_a[2]
    ans[1] = vector_a[2]*vector_b[0] - vector_b[2]*vector_a[0]
    ans[2] = vector_a[0]*vector_b[1] - vector_b[0]*vector_a[1]
    return ans


def rotateVector(vector: numpy.ndarray, degree: float) -> numpy.ndarray:
    ans = numpy.zeros((2))
    theda = math.radians(degree)
    ans[0] = math.cos(theda) * vector[0] - math.sin(theda) * vector[1]
    ans[1] = math.sin(theda) * vector[0] + math.cos(theda) * vector[1]
    return ans