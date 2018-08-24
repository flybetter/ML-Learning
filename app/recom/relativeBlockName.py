# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app import approot
import json


# 临时存储文件
def readCSV(filename):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)
    data = pd.read_csv(filename)
    data.columns = ['id', 'name', 'parent_json', 'children_json']
    blockname = pd.DataFrame(columns=['name', 'blockname', 'count'])
    for index, row in data.iterrows():
        objects = json.loads(row['children_json'])
        print((row['id'], row['name']))
        for object in objects:
            blockname = blockname.append(
                pd.DataFrame({"name": row['name'], "blockname": object['name'], "count": object['count']}, index=["0"]),
                ignore_index=True)
    print(blockname)
    blockFile = approot.get_dataset("blocknames.csv")
    blockname.to_csv(blockFile)


# 读取文档排序后保存
def readCSV2(filename):
    data = pd.read_csv(filename)
    dataGroup = data.groupby("name")
    result = dict()
    tempdict = dict()
    for name, groupstack in dataGroup:
        tempdict.clear()
        temp = groupstack.sort_values(by="count", ascending=False)
        for index, row in temp.iterrows():
            tempdict[row["blockname"]] = row["count"]
            result[name] = tempdict.copy()

    filename = approot.get_dataset("relativeBlockName.json")
    file = open(filename, 'w', encoding='utf-8')
    json.dump(result, file, ensure_ascii=False)


if __name__ == '__main__':
    # filename = approot.get_dataset("SELECT_t___FROM_demo_crawlWeight2_t.csv")
    # readCSV(filename)
    filename = approot.get_dataset("blocknames.csv")
    readCSV2(filename)
