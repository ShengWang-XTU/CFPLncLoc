
import numpy as np


def AvgF1(y_pre, y_true):
    """ Calculation of the results of the evaluation metric AvgF1 """
    total = 0
    p_total = 0
    p, r = 0, 0
    for yt, yp in zip(y_true, y_pre):
        ytNum = sum(yt)
        if ytNum == 0:
            continue
        rec = sum(yp[yt == 1]) / ytNum
        r += rec
        total += 1
        ypSum = sum(yp)
        if ypSum > 0:
            p_total += 1
            pre = sum(yt[yp == True]) / ypSum
            p += pre
    r /= total
    if p_total > 0:
        p /= p_total
    return 2 * r * p / (r + p)


def PrecisionInTop(Y_prob_pre, Y, n):
    """ Calculation of the results of the evaluation metric P@1 """
    Y_pre = np.argsort(1 - Y_prob_pre, axis=1)[:, :n]
    return sum([sum(y[yp]) for yp, y in zip(Y_pre, Y)]) / (len(Y) * n)
