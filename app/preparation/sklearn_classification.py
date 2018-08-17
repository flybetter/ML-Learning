# coding=utf-8
from sklearn import linear_model
import matplotlib.pyplot as  plt


def linerRegression():
    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    print(reg.coef_)
    plt.scatter([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    plt.show()


def Ridge():
    reg = linear_model.Ridge(alpha=.5)
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, 1, 1])
    print(reg.coef_)
    print(reg.intercept_)


if __name__ == '__main__':
    Ridge()
