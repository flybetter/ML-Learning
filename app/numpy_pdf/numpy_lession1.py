# coding=utf-8
import numpy as np


def lesson_1():
    x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], ])
    print(x)
    rows = np.array([[0, 0], [3, 3]])
    cols = np.array([[0, 2], [0, 2]])
    y = x[rows, cols]
    print(y)


if __name__ == '__main__':
    lesson_1()
