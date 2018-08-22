import numpy as np
import pandas as pd
from app import approot


def first():
    s = pd.Series([1, 3, 6, np.nan, 44, 1])
    print(s)
    dates = pd.date_range('20180811', periods=6)
    print(dates)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
    print(df)

    df = pd.DataFrame(np.arange(12).reshape((3, 4)))
    print(df)

    df = pd.DataFrame({'A': 1.,
                       'B': pd.Timestamp('20130102'),
                       'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                       'D': np.array([3] * 4, dtype='int32'),
                       'E': pd.Categorical(["test", "train", "test", "train"]),
                       'F': 'foo'})
    print(df)
    print(df.index)
    print(df.columns)
    print(df.values)
    print(df.describe())
    print(df.dtypes)
    print(np.transpose(df))
    print(df.T)
    print(df.sort_index(axis=1, ascending=False))
    print(df.sort_values(by='E'))


def second():
    dates = pd.date_range('20180811', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
    print(df)
    print(df['A'], df.A)
    print(df[0:3])

    # select by lable:loc

    print(df.loc['20180812'])
    print(df.loc[:, ['A', 'B']])

    # select by position:iloc
    print(df.iloc[3:5, 1:2])
    print(df.iloc[[1, 3, 5], 1:3])

    # mixed selection:ix
    # print(df.ix[:3, ['A', 'C']])

    # Boolean indexing
    print(df[df['A'] < 8])


def third():
    dates = pd.date_range('20180810', periods=6)

    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

    print(df)

    df.iloc[2, 2] = 111
    df.loc['20180810', 'B'] = 2222

    df.A[df.A > 0] = 0
    print(df)

    df['F'] = np.nan
    print(df)
    df['E'] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20180810', periods=6))
    print(df)


def four():
    dates = pd.date_range('20180810', periods=6)

    df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

    print(df)

    df.iloc[0, 1] = np.nan
    df.iloc[1, 2] = np.nan
    print(df)

    print(df.dropna(axis=0, how='any'))  # how={'any','all'} any 是有一个nan就丢掉 all是所有都是nan才丢掉

    print(df.fillna(value=0))
    print(np.any(df.isnull() == True))


def import_export():
    path = approot.get_dataset('Data.csv')

    data = pd.read_csv(path)
    print(data)

    path = approot.get_dataset('Data.pickle')
    data.to_pickle(path)


def concat():
    ##concatenating
    df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
    df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])

    print(df1)
    print(df2)
    print(df3)

    res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

    print(res)

    # join,['inner','outer']

    df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
    df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])

    print(df1)
    print(df2)
    res = pd.concat([df1, df2], join='inner', ignore_index=True)
    # outer就是在外面插入nan，inner就是裁剪不同的nan
    print(res)
    print(pd.concat([df1, df2], axis=1, join_axes=[df1.index]))

    df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
    df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
    df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
    res = df1.append([df2, df3], ignore_index=True)
    print(res)

    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    res = res.append(s1, ignore_index=True)
    print(res)


def merge():
    # merging two df by key/keys.(may be used in database)
    # simple example

    left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                         'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})

    right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})

    print(left)
    print(right)

    res = pd.merge(left, right, on='key')
    print(res)

    # consider two keys
    left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                         'key2': ['K0', 'K1', 'K0', 'K1'],
                         'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})

    right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                          'key2': ['K0', 'K0', 'K0', 'K0'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})

    print(left)
    print(right)

    # how=['left','right','outer','inner']
    res = pd.merge(left, right, on=['key1', 'key2'], how='outer')
    print(res)

    # merging two df key/keys.(may be used in database)
    # indicator

    df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
    df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})

    print(df1)
    print(df2)
    res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
    res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
    print(res)

    # merge by index
    left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                         'B': ['B0', 'B1', 'B2']},
                        index=['K0', 'K1', 'K2'])

    right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                          'D': ['D0', 'D2', 'D3']},
                         index=['K0', 'K2', 'K3'])

    print(left)
    print(right)

    # left_index and right_index

    res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
    print(res)
    res = pd.merge(left, right, left_index=True, right_index=True, how='inner')
    print(res)

    boys = pd.DataFrame({"k": ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
    girls = pd.DataFrame({'k': ['K0', 'K1', 'K3'], 'age': [4, 5, 6]})

    print(boys)
    print(girls)

    res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
    print(res)


if __name__ == '__main__':
    import_export()
