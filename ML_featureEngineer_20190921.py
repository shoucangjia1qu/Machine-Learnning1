# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:22:11 2019

@author: ecupl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("D:\mywork")

#一、过滤式选择
##Relief(Relevant Features)方法
class Relevant_feature(object):
    #0、属性
    def __init__(self):
        self.nearArray = 0          #储存样本对应猜对近邻核猜错近邻的数组
        self.W = 0                  #统计量
        self.trainSet = 0           #原始数据集
        self.normSet = 0            #[0,1]规范化后的数据集
        self.yLabel = 0             #数据标签
        self.labelSet = 0           #标签种类集合
        self.setPercent = 0         #刨去本类数据后，其他各类数据的占比

    
    #1、初始化参数
    def initParas(self, X, Y):
        """
        input：数据集、分类标签
        action：规范化数据、保存分类标签集合、求各类数据占比
        """
        self.trainSet = X
        self.yLabel = Y
        self.normSet = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
        self.labelSet, self.setPercent = self.calSetPercent(Y)
        return
        
    
    #2、计算除本类数据外其他数据的占比
    def calSetPercent(self, Y):
        """
        输入：分类标签
        输出：分类标签的值集合、除去自身类，其他类的占比
        """
        labelSet = list(np.unique(Y))
        if len(labelSet) == 2:
            setPercent = np.ones((2,2))
        elif len(labelSet) < 2:
            raise ValueError("样本分类少于2类")
        else:
            setPercent = np.ones((len(labelSet), len(labelSet)))
            for idxI, yValueI in enumerate(labelSet):
                not_yValueI = sum(Y != yValueI)
                for idxJ, yValueJ in enumerate(labelSet):
                    if idxI == idxJ:
                        continue
                    setPercent[idxI, idxJ] = sum(Y == yValueJ)/not_yValueI
        return labelSet, setPercent

    
    #3、计算每个样本对应每个类别的最近邻样本
    def calNearArray(self):
        """
        计算每个样本对应每个类别中最近邻的样本，
        行代表每个样本，
        列代表每个类别，
        值代表最近邻样本的下标。
        """
        m, d = self.trainSet.shape
        n = len(self.labelSet)
        nearArray = np.zeros((m, n))
        for y_Idx, yValue in enumerate(self.labelSet):
            newxIdx = list(np.nonzero(self.yLabel == yValue)[0])
            newxArr = self.trainSet[newxIdx]
            for xi_Idx in range(m):
                xi = self.trainSet[xi_Idx]
                if xi_Idx in newxIdx:
                    self_Idx = newxIdx.index(xi_Idx)
                    nearnewxIdx = self.edist(xi, newxArr, selfIdx=self_Idx)
                else:
                    nearnewxIdx = self.edist(xi, newxArr)
                nearArray[xi_Idx, y_Idx] = newxIdx[nearnewxIdx]
        self.nearArray = nearArray
        return 
            
            
    #4、计算欧式距离，并返回最小距离的下标
    def edist(self, v1, v2, selfIdx=None):
        dist = np.linalg.norm((v1-v2), axis=1)
        if selfIdx is not None:
            dist[selfIdx] = np.inf
        return dist.argmin()


    #5、求最终变量的相关量
    def train(self, X, Y):
        self.initParas(X, Y)            #初始化参数
        self.calNearArray()             #求每个样本每类的最近邻样本
        m, d = X.shape
        W = np.zeros(d)                 #初始化相关量
        for column in range(d):
            ###这里要加一个变量属性的判断
            ###连续变量
            ###离散变量
            colStyle = 'continuous'
            for row in range(m):
                wi = self.rfValue(row, column, marker=colStyle)
                W[column] += wi
        self.W = W/m
        return
    
    
    #6、求单个样本、单个属性对应的相关量
    def rfValue(self, row, column, marker='continuous'):
        rf = np.zeros(len(self.labelSet))
        xi = self.normSet[row, column]                      #xi自身对应的属性值
        yIdx = self.labelSet.index(self.yLabel[row])        #xi对应分类的下标
        if marker == 'continuous':
            for near_idx, near_ylabel in enumerate(self.labelSet):
                near = int(self.nearArray[row, near_idx])   #xi的近邻样本下标
                xi_near = self.normSet[near, column]        #xi近邻对应的属性值
                if near_idx == yIdx:
                    rf_hit = -np.power((xi-xi_near), 2)     #xi猜中近邻的值
                    rf[near_idx] = rf_hit
                else:
                    rf_miss = np.power((xi-xi_near), 2)     #xi猜错近邻的值
                    rf[near_idx] = rf_miss
        else:
            for near_idx, near_ylabel in enumerate(self.labelSet):
                near = int(self.nearArray[row, near_idx])            
                xi_near = self.normSet[near, column]        
                if near_idx == yIdx:
                    if xi == xi_near:
                        rf_hit = 0                          #离散变量中xi和猜中近邻值相同时，为0
                    else:
                        rf_hit = -1                         #离散变量中xi和猜中近邻值不相同时，为-1
                    rf[near_idx] = rf_hit
                else:
                    if xi == xi_near:
                        rf_miss = 0                         #离散变量中xi和猜错近邻值相同时，为0
                    else:
                        rf_miss = 1                         #离散变量中xi和猜错近邻值不相同时，为1                        
                    rf[near_idx] = rf_miss
        return sum(np.multiply(rf, self.setPercent[yIdx]))
            
###训练测试
if __name__ == "__main__":
    relief = Relevant_feature()
    relief.train(X, Y)
    W = relief.W                        #各属性的统计量
    normX = relief.normSet              #[0,1]规范化后的数据集
    nearArray = relief.nearArray        #各个样本各类别中的近邻样本下标


###验算1：求最近邻样本
ySet = list(np.unique(Y))
nearArr = np.zeros((m,3))
for i in range(m):
    xi = X[i]
    for d in range(3):
        minDist = np.inf
        minIdx = 0
        xjSet = np.nonzero(Y==ySet[d])[0]
        for j in xjSet:
            if i == j:
                continue
            else:
                xj = X[j]
                dist = np.linalg.norm(xi-xj)
                if dist < minDist:
                    minDist = dist
                    minIdx = j
        nearArr[i, d] = minIdx


###验算2：求统计量
m, d = X.shape
#col = 0
for col in range(4):
    rf = 0
    for i in range(m):
        for j in range(3):
            if (i<50 and j==0) or (i>=50 and i<100 and j==1) or (i>=100 and j==2):
                rf -= np.power((normX[i,col] - normX[int(nearArray[i,j]),col]), 2)
            else:
                rf += 0.5*np.power((normX[i,col] - normX[int(nearArray[i,j]),col]), 2)
    rf/=m
    print(rf)


###西瓜集数据测试
from sklearn.preprocessing import OrdinalEncoder
dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
    ]
#特征值列表
labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
dataX = np.array(dataSet)[:,:6]
oriencode = OrdinalEncoder(categories='auto')
oriencode.fit(dataX)
X1=oriencode.transform(dataX)           #编码后的数据
X2=np.array(dataSet)[:,6:8].astype(float)
X = np.hstack((X1,X2))
Y = np.array(dataSet)[:,8]
Y[Y=="好瓜"]=1
Y[Y=="坏瓜"]=0
Y=Y.astype(float)
Y = Y.reshape(-1,1)

relief = Relevant_feature()
relief.train(X, Y)
W = relief.W                        #各属性的统计量


###另编一个，按照顺序来的
#1、求K近邻函数
def K_near(xi, Xset, K=1, selfIdx=None):
    distList = np.linalg.norm((xi-Xset), axis=1)
    if selfIdx is not None:
        distList[selfIdx] = np.inf
    idx_sort = np.argsort(distList)
    return idx_sort[:K]

#2、正式求统计量
def cal_rf(X, Y, column, K=1, col_type="continuous"):
    m, d = np.shape(X)
    Yset = list(np.unique(Y))               #分类标签的集合
    W = 0                                   #初始化统计量
    normX = X[:,column]                     #将属性值进行规范化，离散变量不需要，连续变量需要
    if col_type=="continuous":
        normX = (normX-normX.min())/(normX.max()-normX.min())
    for i in range(m):
        xi = X[i]                           #样本
        yi = Y[i]                           #样本的标签
        xy_value = normX[i]                 #样本对应的变量值
        for label_idx, label in enumerate(Yset):
            Xset_idx = np.nonzero(Y==label)[0]
            Xset_idx_list = list(np.nonzero(Y==label)[0])
            Xset = X[Xset_idx,:]
            if label==yi:
                self_idx = Xset_idx_list.index(i)
                nhit_idx = K_near(xi, Xset, K, self_idx)
                nhit_set = normX[Xset_idx[nhit_idx]]      #同样本类型的变量值
                if col_type=="continuous":
                    if K>1:
                        W = W - sum(np.power((xy_value-nhit_set),2))
                    else:
                        W = W - np.power((xy_value-nhit_set),2)
                else:
                    for nhit_i in nhit_set:
                        if nhit_i != xy_value:
                            W -= 1
                        else:
                            pass
            else:
                pi = sum(Y==label)/(len(Y)-sum(Y==yi))
                nmiss_idx = K_near(xi, Xset, K)
                mhit_set = normX[Xset_idx[nmiss_idx]]
                if col_type=="continuous":
                    if K>1:
                        W += pi*sum(np.power((xy_value-mhit_set),2))
                    else:
                        W += pi*np.power((xy_value-mhit_set),2)
                else:
                    for mhit_i in mhit_set:
                        if mhit_i != xy_value:
                            W += pi*1
                        else:
                            pass
    W = W/(m*K)
    return W
    
for i in range(4):
    rf = cal_rf(X, Y, i)
    print(rf)








