# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def first():
    x = np.linspace(-1, 1, 50)
    y = x ** 2 + 1
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    first()
