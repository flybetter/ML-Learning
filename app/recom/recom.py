# coding=utf-8
from app import approot
import pandas as pd
import numpy as np
import os
from wordcloud import WordCloud
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from urllib import request
import re
import logging
import json
import matplotlib.font_manager as fm

logging.basicConfig(level=logging.DEBUG)

ESF_URL = 'http://mapi.house365.com/taofang/v1.0/esf/?method=getHouseListNew&name=HouseSellEX&city=nj&id='


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
    data = pd.read_csv(file, usecols=[0, 1])
    data = data.dropna(thresh=2)
    data.columns = ['deviceId', 'contentId']
    data = data.groupby('deviceId')
    # value = data.get_group('865584030908872')
    # return value.iloc[:, 1].dropna(how='any'), value.iloc[:, 2].dropna(how='any')
    for i, (name, group) in enumerate(data):
        # if i == 100:
        #     break
        contentIds = getContentNum(group.iloc[:, 1].dropna(how='any'))
        df = searchUrl(contentIds)
        if not df.empty:
            drawPicture(df, name)


def wordCloudDemo(words):
    # file = approot.get_root('constitution.txt')
    cloud = WordCloud(
        # 设置字体，不指定就会出现乱码
        font_path="HYQiHei-25J.ttf",
    )
    word = cloud.generate(words)
    plt.imshow(word)
    plt.axis('off')
    plt.title('search key world cloud')
    plt.show()


def getContentNum(contentId):
    result = '111'
    for content in contentId:
        if re.match(r'\d+|1-\d+', str(content)):
            result += ',' + re.match(r'(1-)?(\d+)', str(content)).group(2)
    return result


def searchUrl(contentIds):
    df = pd.DataFrame(columns=['id', 'price', 'blockname'], dtype=np.int8)
    logging.debug(ESF_URL + contentIds)

    for i in range(0, len(contentIds), 350):
        tempContentIds = contentIds[i:i + 350]
        response = request.urlopen(ESF_URL + tempContentIds)
        response = response.read().decode('utf-8')
        logging.info(response)
        if response != '-1':
            jsonObjects = json.loads(response)
            for object in jsonObjects:
                df = df.append(pd.Series([object['id'], float(object['price']), object['blockinfo']['blockname']],
                                         index=list(df.columns)), ignore_index=True)
    return df


def drawPicture(df, name):
    # 用subplot()方法绘制多幅图形
    plt.figure(figsize=(12, 24), dpi=80)
    # 创建第一个画板
    plt.figure(1)
    # 将第一个画板划分为2行1列组成的区块，并获取到第一块区域
    ax1 = plt.subplot(411)
    ax1.clear()

    # 在第一个子区域中绘图
    one = df[df.price < 100].count()
    one2two = df[(100 <= df.price) & (df.price < 200)].count()
    two2three = df[(200 <= df.price) & (df.price < 300)].count()
    three2four = df[(300 <= df.price) & (df.price < 400)].count()
    four2five = df[(400 <= df.price) & (df.price < 500)].count()
    five2six = df[(500 <= df.price) & (df.price < 600)].count()
    six2seven = df[(600 <= df.price) & (df.price < 700)].count()
    seven2eight = df[(700 <= df.price) & (df.price < 800)].count()
    eight2nine = df[(800 <= df.price) & (df.price < 900)].count()

    labels = (
        '100', '100~200', '200~300', '300~400', '400~500', '500~600', '600~700', '700~800', '800~900')
    sizes = [one.price, one2two.price, two2three.price, three2four.price, four2five.price, five2six.price,
             six2seven.price, seven2eight.price, eight2nine.price]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('price pie chart')

    # 选中第二个子区域，并绘图
    ax2 = plt.subplot(412)

    cloud = WordCloud(
        # 设置字体，不指定就会出现乱码
        font_path="HYQiHei-25J.ttf",
        background_color="white",
    )
    blockName = df.iloc[:, 2]
    word = cloud.generate(" ".join(blockName))
    plt.imshow(word)
    plt.axis('off')

    ax4 = plt.subplot(413)
    ax4.clear()
    font = fm.FontProperties(fname='HYQiHei-25J.ttf')
    dfgroup = df.groupby('blockname')
    name_list = list()
    num_list = list()
    for blockName, group in dfgroup:
        name_list.append(blockName)
        num_list.append(len(group))

    plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
    plt.xticks(fontproperties=font)

    ax3 = plt.subplot(414)
    ax3.clear()
    plt.plot(np.arange(df['price'].count()), df.iloc[:, 1], marker='o', mfc='w')

    # 为第一个画板的第一个区域添加标题
    ax1.set_title("price range")
    ax2.set_title("block name word cloud")
    ax3.set_title("price broken line")
    ax4.set_title("block visit count")

    # 调整每隔子图之间的距离
    plt.tight_layout()
    # 添加关联关系
    name = relation(str(name))
    filename = approot.get_picture(str(name) + '.jpg')
    fig = plt.gcf()
    plt.show()
    fig.savefig(filename)


def test():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def test2():
    # 用subplot()方法绘制多幅图形
    plt.figure(figsize=(12, 12), dpi=80)
    # 创建第一个画板
    plt.figure(1)
    # 将第一个画板划分为2行1列组成的区块，并获取到第一块区域
    ax1 = plt.subplot(211)

    # 在第一个子区域中绘图
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # 选中第二个子区域，并绘图
    ax2 = plt.subplot(212)
    plt.plot([2, 4, 6], [7, 9, 15])

    plt.figure(1)

    # 为第一个画板的第一个区域添加标题
    ax1.set_title("第一个画板中第一个区域")
    ax2.set_title("第一个画板中第二个区域")

    # 调整每隔子图之间的距离
    plt.tight_layout()
    plt.show()


def relation(deviceId):
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

    flag = "nothing"
    for key, value in relations.items():
        if value == deviceId:
            flag = key
            break
    return flag


def test3():
    font = fm.FontProperties(fname='HYQiHei-25J.ttf')

    name_list = ['星期一', '星期二', '星期三', '星期四']
    num_list = [1.5, 0.6, 7.8, 6]
    plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
    plt.xticks(fontproperties=font)
    plt.show()


if __name__ == '__main__':
    # select_DEVICE_ID_CONTEXT_ID_from_DWB_DA_8_27.csv  设备id 864621038192553 的行为记录

    file = approot.get_dataset('select_DEVICE_ID_CONTEXT_ID__from_DWB_DA.csv')
    readCsv(file=file)
    # wordCloudDemo(' '.join(searchKey.values))
    # contentIds = getContentNum(contentId)
    # df = searchUrl(contentIds)
    # drawPicture(df)
    # test2()
    # test()
    # test3()
