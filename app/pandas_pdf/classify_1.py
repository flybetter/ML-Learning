6  # coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_object():
    data = pd.Series([1, 2, 3, 4, np.nan])
    print(data)

    date = pd.date_range(start='20180801', periods=6)
    print(date)

    df = pd.DataFrame(np.random.randn(6, 4), index=date, columns=list("ABCD"))
    print(df)

    df2 = pd.DataFrame({"A": 1,
                        "B": pd.Timestamp('20130102'),
                        "C": pd.Series(1, index=list(np.arange(5)), dtype='float32'),
                        "D": np.array([3] * 5, dtype="int32"),
                        "E": pd.Categorical(["test", "train", "test", "train", "test"]),
                        "F": "foo"})

    print(df2)
    print(df2.dtypes)


def query_data():
    df = pd.DataFrame(np.random.randn(5, 4), index=pd.date_range("20180801", periods=5), columns=list("ABCD"))
    print(df)

    print(df.head())
    print(df.tail(2))

    print(df.index)
    print(df.columns)
    print(df.values)
    print(df.describe())
    print(df.T)

    print(df.sort_index(ascending=False))

    print(df.sort_index(by="B", ascending=False))


def select():
    dates = pd.date_range('20180801', periods=6)

    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
    print(df)

    print(df["A"])

    print(df[0:3])

    print(df['2018-08-03':'2018-08-04'])

    print(df.loc[dates[0]])

    print(df.loc[:, ['A', 'B']])

    print(df.loc['2018-08-03':'2018-08-04', ['A', 'B']])

    print(df.loc[dates[0], 'A'])

    print(df.iloc[3])
    print(df.iloc[1:3, 2:4])

    print(df[df.A > 0])

    print(df[df > 0])

    df2 = df.copy()

    df2['E'] = ['one', 'two', 'three', 'four', 'five', 'six']

    print(df2)

    print(df2[df2['E'].isin(['one', 'two'])])


def add_new_column():
    dates = pd.date_range('20180801', periods=6)

    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

    data = pd.Series(np.arange(6), index=dates)

    print(data)

    df['E'] = data

    print(df)

    df.loc[dates[0], 'A'] = 0

    print(df)

    df.loc[:, 'D'] = np.array([5] * len(df))

    print(df)


def deal_default_value():
    # reindex 方法可以对于轴上的索引进行增加/修改/删除的操作

    dates = pd.date_range('20180801', periods=6)

    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

    df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])

    df1.loc[dates[0]:dates[1], ['E']] = 1

    print(df1)

    print(df1.dropna(how='any'))

    print(df1.fillna(5))

    print(pd.isnull(df1))


def the_relevant_operation():
    df = pd.DataFrame(np.random.randn(6, 4), index=pd.date_range('20180801', periods=6), columns=list('ABCD'))

    print(df)

    print(df.mean())

    print(df.mean(1))

    s = pd.Series(np.arange(6), index=pd.date_range('20180801', periods=6)).shift(2)
    print(s)

    print(df.sub(s, axis='index'))

    print(df.apply(np.cumsum, axis=1))

    print(df.apply(lambda x: x.max() - x.min()))


def action():
    # concat
    df = pd.DataFrame(np.random.randn(10, 4))

    pieces = [df[:3], df[3:7], df[7:]]

    print(pd.concat(pieces))

    # merge的操作直接看官网的说明文档就好

    # Append 也直接看文档就比较好


def group_by():
    # 对于group by操作，我们通常指一下一个或者多个操作步骤
    # spitting按照一些规则将数据分为不同的组
    # applying对于每组数据分别执行一个函数
    # combining 将结果组合到一个数据结构中
    df = pd.DataFrame({"A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "bar"],
                       "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                       "C": np.random.randn(8),
                       "D": np.random.randn(8)})

    print(df.groupby("A").sum())
    print(df.groupby(["A", "B"]).sum())


def change_form():
    tuples = tuple(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                         ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))

    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

    df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
    df2 = df[:4]
    print(df2)
    print(df)
    print(df.stack())


if __name__ == '__main__':
    change_form()
