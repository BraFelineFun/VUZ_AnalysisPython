import numpy as np
import random


def getValues(iteration):
    x = np.random.standard_t(5)  # student
    e = np.random.normal(0, 1)  # normal
    y = 3 * x + 5 + e

    print("{3:9} | {0:8.3f} {1:8.3f} {2:8.3f} |\n".format(y, x, e, iteration))

    return y, x


def execute():
    N = 10
    Y = []
    X = []
    print("iteration |     y        x        e     |\n")
    for i in range(N):
        values = getValues(i)
        Y.append(values[0])
        X.append(values[1])

    X_modified = np.c_[np.ones(N), np.vstack(X)]

    B = np.linalg.inv(np.transpose(X_modified).dot(X_modified)).dot(np.transpose(X_modified).dot(Y))
    print("B is  = ", B, "\n")

    estimates = []
    for i in range(N):
        estimates.append(B[0] + B[1] * X[i])

    print("Estimates are: \n", estimates)
