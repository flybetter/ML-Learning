# coding=utf-8
import numpy as np
import pandas as pd
from app import approot
from app.recom.recom import searchUrl, getContentNum
import json
from app.recom import oracle_connect


# 读取历史记录
def read_history(filePath):
    pd.set_option('display.width', 300)  # 设置字符显示宽度
    pd.set_option('display.max_rows', None)  # 设置显示最大行
    pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列
    data = pd.read_csv(filePath, usecols=[0, 1])
    data = data.dropna(thresh=2)
    data.columns = ['deviceId', 'contentId']
    data = data.groupby('deviceId')
    # value = data.get_group('865584030908872')
    # return value.iloc[:, 1].dropna(how='any'), value.iloc[:, 2].dropna(how='any')
    for i, (name, group) in enumerate(data):
        contentIds = getContentNum(group.iloc[:, 1].dropna(how='any'))
        print("deviceId:"+str(name))
        df = searchUrl(contentIds)
        if not df.empty:
            statistical(df, name)


def statistical(df, name):
    relativeBlocksName = list()
    blocknames = df.groupby(by="blockname")
    for name, block in blocknames:
        relativeBlocksName.extend(relative_blocks(name))
    recomm_datas = oracle_connect.get_data(relativeBlocksName, 200, 300)


def price_param():
    pass


def block_param():
    pass


def relative_blocks(blockname):
    relative_block_name = list()
    relativeBlockFile = approot.get_dataset("relativeBlockName.json")
    file = open(relativeBlockFile, 'r', encoding='utf-8')
    jsonObject = json.load(file)
    if blockname in jsonObject.keys():
        for key in jsonObject[blockname]:
            relative_block_name.append(key)
    return relative_block_name


if __name__ == '__main__':
    filename = approot.get_dataset("select_DEVICE_ID_CONTEXT_ID_from_DWB_DA_8_27.csv")
    read_history(filename)
