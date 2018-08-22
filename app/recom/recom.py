# coding=utf-8
from app import approot
import pandas as pd
import numpy as np
import os


def readFile(file):
    pd.set_option('display.width', 300)  # 设置字符显示宽度
    pd.set_option('display.max_rows', None)  # 设置显示最大行
    pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列
    with open(file, 'r') as f:
        data = f.readlines()

    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"
    data_df = pd.read_json(data_json_str)
    print(data_df.head)


def readFile2(file):
    data = pd.read_json(file, lines=True)
    print(data.head)


def readCsv(file):
    pd.set_option('display.width', 300)  # 设置字符显示宽度
    pd.set_option('display.max_rows', None)  # 设置显示最大行
    pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列
    data = pd.read_csv(file, usecols=[6, 36, 39])
    data = data.dropna(thresh=2)
    data.columns = ['deviceId', 'searchKey', 'contentId']
    data = data.groupby('deviceId')
    for name, group in data:
        print((name, group))


if __name__ == '__main__':
    file = approot.get_root('select___from_DWB_DA_APP_STATISTICS_wher.csv')
    readCsv(file=file)
