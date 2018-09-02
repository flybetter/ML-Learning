# coding=utf-8
import pandas as pd
import numpy as np


def outlier_demo():
    States = ["NY", "NY", "NY", "NY", "FL", "FL", "GA", "GA", "FL", "FL"]
    data = list(range(1, 11))
    idx = pd.date_range('1/1/2012', periods=10, freq="MS")
    df1 = pd.DataFrame(data=data, index=idx, columns=['Revenue'])
    df1['State'] = States
    data = ['10', '10', '9', '9', '8', '8', '7', '7', '6', '6']
    idx = pd.date_range('1/1/2013', periods=10, freq="MS")
    df2 = pd.DataFrame(data=data, index=idx, columns=['Revenue'])
    df2['State'] = States

    df = pd.concat([df1, df2])
    newdf = df.copy()
    newdf['Revenue'] = newdf['Revenue'].astype(float)
    newdf['x-Mean'] = abs(newdf['Revenue'] - newdf['Revenue'].mean())
    newdf['1.96*std'] = 1.96 * newdf['Revenue'].std()
    newdf['Outlier'] = abs(newdf['Revenue'] - newdf['Revenue'].mean()) > 1.96 * newdf['Revenue'].std()

    # print(newdf)

    # second way
    newdf = df.copy()
    print(newdf)
    States = newdf.groupby('State')

    newdf['Revenue'] = newdf['Revenue'].astype(float)

    newdf['Outlier'] = States.transform(lambda x: abs(x - x.mean()) > 1.96 * x.std())
    newdf['x-Mean'] = States.transform(lambda x: abs(x - x.mean()))
    newdf['1.96*std'] = States.transform(lambda x: 1.96 * x.std())
    print(newdf)




if __name__ == '__main__':
    outlier_demo()
