# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 21:47:04 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
from collections import Iterable
import os

os.chdir(r"D:mywork\test")

###关联算法，Apriori算法
data = [[1,2,5], [2,4], [2,3], [1,2,4], [1,3], [2,3], [1,3], [1,2,3,5], [1,2,3]]

class Apriori(object):
    #属性
    def __init__(self):
        self.ItemSet = []                   #每组生成的关联物品组
        self.ItemSupport = []               #每组物品组的支持度
        self.ItemConfident = []             #每组物品组的置信度
        self.SupportThres = 0               #置信度的阀值
        self.CutSet = []                    #每轮循环剪枝的物品组
        self.X = 0                          #训练数据集
        self.row = 0                        #物品样本数
        self.AssoRules = {}                 #最终生成的关联规则
    
    #判断x是否X的子集
    def judgeSubset(self, x, Xset):
        if x.issubset(Xset):
            return 1
        else: 
            return 0
    
    #初始化数据集，变为C1，每个物品变成一个物品组
    def initSet(self, Xset, supthres):
        self.X = Xset
        self.row = len(Xset)
        self.SupportThres = supthres
        C1 = []
        for items in Xset:
            for i in items:
                if i not in C1:
                    C1.append(i)
        C1.sort()                               #数据集从小到大排序
        C = self.changeFS(C1)
        self.ItemSet.append(C)                  #第一步加入数据集
        supportDict, cutDict = self.calSupport(C)
        self.ItemSupport.append(supportDict)    #第二步加入符合支持度的数据集
        self.CutSet.append(cutDict)             #第三步加入不符合支持度的数据集
        return 
        
    #将列表中的元素变成frozenset
    def changeFS(self, li):
        fslist = []
        for i in li:
            if not isinstance(i, Iterable):               #判断是否可循环数据类型
                i = [i]
            fslist.append(frozenset(i))
        return fslist

    #计算支持度，符合阀值的留下，低于的去掉
    def calSupport(self, Ci):
        supportDict = {}
        cutDict = {}
        for subset in Ci:
            hit = 0
            for subX in self.X:
                if self.judgeSubset(subset, subX) == 1:
                    hit += 1
            supRatio = hit/self.row                             #计算支持率
            if supRatio>=self.SupportThres:
                supportDict[subset] = supRatio 
            else: 
                cutDict[subset] = supRatio
        return supportDict, cutDict
    
    #生成新的数组
    def add(self, k):
        LCi = []
        data = self.ItemSupport[k]
        for fs in data.keys():
            lastfs = list(fs)[:-1]
            for subfs in data.keys():
                if fs == subfs:
                    continue
                lastsubfs = list(subfs)[:-1]
                if lastfs != lastsubfs:
                    continue
                newFS = fs.union(subfs)
                lfs = list(newFS)                               #将frozenset转换为列表
                lfs.sort()                                      #并排序
                if lfs in LCi:
                    continue
                LCi.append(lfs)                                 #C1数据生成C2数据，并重新排序
        Ci = self.changeFS(LCi)                                 #将每个数据组变成frozenset格式
        self.ItemSet.append(Ci)                                 #第一步加入数据集
        supportDict, cutDict = self.calSupport(Ci)              
        self.ItemSupport.append(supportDict)                    #第二步加入符合支持度的数据集
        self.CutSet.append(cutDict)                             #第三步加入不符合支持度的数据集
        return
    
    #对数据集进行剪切
    def cut(self, k):
        cutlist = []                                            #存放需要剪枝的数据集
        addcutdict = {}                                         #存放剪枝需要增加的数据集字典
        cutdata = self.CutSet[k]
        supportdata = self.ItemSupport[k+1]
        for cutx in cutdata.keys():
            for supx, supratio in supportdata.items():
                if cutx.issubset(supx):
                    cutlist.append(cutx)
                    addcutdict[supx] = supratio
        self.CutSet[k+1].update(addcutdict)                     #剪枝下来的数据集字典放入剪枝集中
        self.cutleaf(k+1, cutlist)                              #根据简直列表的元素进行剪枝
        return
    
    #根据剪枝列表中的清单进行剪枝
    def cutleaf(self, k, cutList):
        for i in cutList:
            self.ItemSupport[k].remove(i)
        return
    
    #循环遍历数据集
    def train(self, Xset, supthres):
        self.initSet(Xset, supthres)
        k = 0
        while len(self.ItemSupport[k]) > 0:
            self.add(k)                                         #生成新的数据集，支持度不符合的已经被剔除
            self.cut(k)                                         #对新的数据集进行剪切
            k += 1
        return
        

#训练
ap = Apriori()
ap.train(data, 0.2)        
ItemSet = ap.ItemSet
SupportSet = ap.ItemSupport
CutSet = ap.CutSet
print(ItemSet, '\n')
print(SupportSet, '\n') 
print(CutSet)




