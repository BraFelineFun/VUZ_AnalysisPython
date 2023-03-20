import numpy as np
import random


def strToInt(inp):
    try:
        x = int(inp)
        return x
    except Exception:
        raise Exception("int expected")


def getRndCoef():
    initRand = random.random() - 0.5
    return round(initRand * 100, 1)


def poly():
    rank = strToInt(input("Enter rank of polynomial: "))
    coef = np.array([getRndCoef() for _ in range(rank)])
    print("coefficients = " + str(coef) + "\n")

    roots = np.roots(coef)
    print("roots = " + str(roots))


def getRealRoots():
    rank = strToInt(input("Enter rank of polynomial: "))
    roots = np.array([getRndCoef() for _ in range(rank - 1)])
    coefs = np.poly(roots)
    print(coefs)
