# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lession_second():
    names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
    random_names = list(names[np.random.randint(low=0, high=len(names))] for _ in range(1000))
    print(random_names)
    births = [np.random.randint(low=0, high=1000) for _ in range(1000)]
    print(births)

    BabyDateSet = list(zip(random_names, births))
    print(BabyDateSet[:10])

    df = pd.DataFrame(data=BabyDateSet, columns=["Name", "Births"])
    print(df)

    df.to_csv("birth1880.csv", index=False, header=False)

    print(df.groupby('Name').sum())


if __name__ == '__main__':
    lession_second()
