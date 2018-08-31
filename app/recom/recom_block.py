# coding=utf-8
import numpy as np
import pandas as pd
from app import approot
from app.recom.recom import searchUrl, getContentNum
import json
from app.recom import oracle_connect
import logging

logging.basicConfig(level=logging.DEBUG)


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
        logging.debug(name)
        contentIds = getContentNum(group.iloc[:, 1].dropna(how='any'))
        df = searchUrl(contentIds)
        logging.debug("deviceID:" + name)
        if not df.empty:
            statistical(df, name)


# 读取历史记录
def read_history2(filePath, deviceId):
    pd.set_option('display.width', 300)  # 设置字符显示宽度
    pd.set_option('display.max_rows', None)  # 设置显示最大行
    pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列
    data = pd.read_csv(filePath, usecols=[0, 1])
    data = data.dropna(thresh=2)
    data.columns = ['deviceId', 'contentId']
    data = data[data['deviceId'] == deviceId]
    contentIds = getContentNum(data.iloc[:, 1].dropna(how='any'))
    df = searchUrl(contentIds)
    logging.debug("deviceID:" + deviceId)
    if not df.empty:
        statistical(df, deviceId)


def statistical(df, name):
    relativeBlocksName = list()
    blocknames = df.groupby(by="blockname")
    for name, block in blocknames:
        logging.debug("blockname:" + name)
        relativeBlocksName.extend(relative_blocks(name))
    if len(relativeBlocksName) > 0:
        oracle_connect.get_data(set(relativeBlocksName), df['price'].mean() - 50, df['price'].mean() + 50)


def relation(phone):
    relations = dict()
    relations["15077827585"] = "864621038192553"
    relations["13770324189"] = "353460084288793"
    relations["13675189197"] = "865970030389108"
    relations["13601901399"] = "860980031132311"
    relations["17855106781"] = "865970034361434"
    relations["15905175211"] = "866533031935839"
    relations["13390901599"] = "861918034499591"
    relations["15951001888"] = "358520088841320"
    relations["15850780069"] = "869885031905924"
    relations["13851499283"] = "867960033998367"
    relations["13401952869"] = "864284035078679"
    relations["18951603156"] = "A00000751717EE"
    relations["18051082210"] = "867455031299405"
    relations["15295123506"] = "352324072851122"
    relations["15105198373"] = "866318037044764"
    relations["13913993926"] = "863184037700430"
    relations["17302584660"] = "864032031775743"
    relations["18055500055"] = "865032030154899"

    return relations[phone]


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


def startup(phone):
    filename = approot.get_dataset("select_DEVICE_ID_CONTEXT_ID__from_DWB_DA.csv")
    deivceId = relation(phone)
    data = read_history2(filename, str(deivceId))
    # read_history(filename)


if __name__ == '__main__':
    # filename = approot.get_dataset("select_DEVICE_ID_CONTEXT_ID__from_DWB_DA.csv")
    # deivceId = relation("13770324189")
    # read_history2(filename, str(deivceId))
    # # read_history(filename)
    startup("13770324189")
