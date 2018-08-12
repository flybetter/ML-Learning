# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def first():
    # plot data
    # Series
    data = pd.Series(np.random.rand(1000), index=np.arange(1000))
    data = data.cumsum()
    print(data)
    data.plot()
    plt.show()

    df = pd.DataFrame(np.random.randn(1000, 4),
                      index=np.arange(1000),
                      columns=list("ABCD"))
    print(df)
    df = df.cumsum()
    print(df)
    df.plot()
    plt.show()

    # plot methods
    # 'bar','his','box','kde','are','scatter'
    ax = df.plot.scatter(x='A', y='B', color='DarkBlue', label='Class 1')
    df.plot.scatter(x='A', y='C', color='DarkGreen', label='CLass 2', ax=ax)
    plt.show()


if __name__ == '__main__':
    first()
