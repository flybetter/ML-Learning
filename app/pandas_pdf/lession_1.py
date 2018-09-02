# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lession_first():
    names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
    births = [968, 155, 77, 578, 973]
    BabyDataSet = list(zip(names, births))
    df = pd.DataFrame(data=BabyDataSet, columns=['Name', 'Births'])
    df.to_csv('birth1880.csv', index=False, header=False)

    df2 = pd.read_csv('birth1880.csv', names=['Name', 'Births'])
    # print(df2)

    df2['Births'].plot(kind='bar')
    plt.show()


if __name__ == '__main__':
    lession_first()
