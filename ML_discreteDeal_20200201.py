# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:51:08 2020

@author: ecupl
"""

import numpy as np
import pandas as pd
import functools
import time


def executeTime(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start = time.clock()
        ret = func(*args, *kw)
        print("%s函数运行时长%fs"%(func.__name__, time.clock()-start))
        return ret
    return wrapper

@executeTime
def discrete(X, Num=5, threshold=0.9):
    """
    此函数用作对离散变量的处理。
    1.若特征值小于等于指定Num。则可以无需处理，直接返回分类结果。
    2.若特征值大于指定Num。
      过滤低饱和度特征的离散值，将每个特征的离散值按照饱和度依次从高到低排序，
      若前TopN的离散值覆盖度已达threshold，则后续离散值归为一类，一共有N+1类。
    函数会判断是否有空值，若无空值，则按以上步骤进行执行；若有空值，则将空值全部填充为-99。
    ——————————————————————————
    Input:
    X:需要输入进行处理的数据。
    Num:特征的离散值的最小个数，超过的话需对该特征的离散值进行合并处理，默认为5。
    threshold:特征的离散值的覆盖度，超过的话需对剩下的离散值进行合并，默认为0.9。
    ——————————————————————————
    Return:
    featureDF:特征中的离散值、对应的数量、对应的分类。
    sortResult:分类结果。
    """
    X2S = pd.Series(np.copy(X))
    X2S.fillna(-99, inplace=True)
    sampleSize = X2S.size                           #样本数量
    X2SSort = X2S.value_counts()
    sortSize = X2SSort.size                         #离散值的个数
    sortList = list(X2SSort.index)                  #离散值的值
    featureDF = pd.DataFrame(X2SSort, columns=['Num'])      #初始化特征的DF
    featureSort = list()                                    #初始化处理过后的分类值
    featurePercent = list()                                 #初始化每个离散值累加的数量百分比
    sortResult = np.zeros(sampleSize)                       #初始化处理过后的Array
    precentSum = .0                                         #初始化每个离散值数量和为0
    if sortSize <= Num:
        for idx, sortValue in enumerate(sortList):
            sortValueIdx = (X2S==sortValue).nonzero()[0]
            sortResult[sortValueIdx] = idx + 1
            featureSort.append(idx + 1)
            precentSum = featurePercent[-1]+len(sortValueIdx)/float(sampleSize) if idx!=0 else len(sortValueIdx)/float(sampleSize)
            featurePercent.append(precentSum)
    else:
        #当离散值的饱和度占比和还没超过阈值时，一类一类分；当超过后则剩下的所有离散值归为一类。
        for idx, sortValue in enumerate(sortList):
            sortValueIdx = (X2S==sortValue).nonzero()[0]
            sortResult[sortValueIdx] = idx + 1
            featureSort.append(idx + 1)
            precentSum = featurePercent[-1] + len(sortValueIdx)/float(sampleSize) if idx!=0 else len(sortValueIdx)/float(sampleSize)
            featurePercent.append(precentSum)
            if precentSum >= threshold:
                break
        #剩下的不足10%的离散值的处理
        valueLeft = idx + 2                         #剩下其他离散值的分类值
        sortListLeft = sortList[idx+1:]             #剩下其他的离散值
        sortResult[X2S.isin(sortListLeft).nonzero()[0]] = valueLeft
        featureSort += [valueLeft]*len(sortListLeft)
        for sortValueLeft in sortListLeft:
            sortValueLeftIdx = (X2S==sortValueLeft).nonzero()[0]
            featurePercent.append(featurePercent[-1] + len(sortValueLeftIdx)/float(sampleSize))
    featureDF['newSort'] = featureSort; featureDF['percent'] = featurePercent
    return featureDF, sortResult










