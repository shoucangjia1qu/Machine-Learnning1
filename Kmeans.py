# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:43:08 2018

@author: ZWD
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os

os.chdir(r"D:\mywork\test\ML_CCB")
with open("4k2_far_data.txt") as f:
    file = f.readlines()
train = np.array([[float(j) for j in i.split()] for i in file ])
train = train[:,1:]

'''定义类：K－means聚类'''
class kmeans(object):
    '''定义属性'''
    def __init__(self):
        self.K = 0          #目标分类个数
        self.keyPoints = 0  #聚类中心点
        self.trainSet = 0   #训练集
        self.labels = []    #分类结果
        self.dists = []     #距离中心点距离
    
    '''定义欧式距离公式'''
    def eDist(self,v1,v2):
        return(np.linalg.norm(v1-v2))
    '''数据标准化'''
    def normlize(self,data):
        m,n = data.shape
        for col in n:
            for row in m:
                normdata[row,col] = (data[row,col]-data[:,col].mean())/data[:,col].std()
    
    '''定义初始中心点'''
    def randomPoints(self,train,K):
        m,n = train.shape
        keyPoints = np.zeros((K,n))
        for i in range(n):
            maxvalue = np.max(train[:,i])
            minvalue = np.min(train[:,i])
            for j in range(K):
                keyPoints[j,i] = minvalue + np.random.rand()*(maxvalue-minvalue)
        return keyPoints
    
    '''定义k-means聚类函数'''
    def train(self,train,K):
        self.K = K
        self.trainSet = train
        m,n = train.shape
        labelList = np.zeros(m)
        distList = np.zeros(m)
        keyPoints = self.randomPoints(train,K)
        flag = True
        while flag:
            flag = False
            for i in range(m):
                dists = []
                dists = [self.eDist(train[i,:],keyPoints[j,:]) for j in range(K)]
                mindist = min(dists)
                minIdx = dists.index(mindist)
                if labelList[i] != minIdx:
                    flag = True
                labelList[i] = minIdx
                distList[i] = mindist
            '''迭代中心点'''
            for rank in range(K):
                #找到分类标签为rank的数据并重新计算中心点
                newData = train[np.nonzero(labelList==rank)[0]]
                keyPoints[rank] = np.mean(newData,axis=0)
        self.keyPoints = keyPoints
        self.labels = labelList
        self.dists = distList


'''正式程序'''
KM = kmeans()
KM.train(train,4)
Labels = KM.labels
keyPts = KM.keyPoints

'''改进:二分类聚类'''
K=4
point0 = np.mean(train,axis=0)
keyPts = []
keyPts.append(point0.tolist())
Labels = np.zeros(len(train))
Dists = np.zeros(len(train))
for p in range(len(train)):
    Dists[p] = KM.eDist(train[p,:],point0)
#设置初始总误差
while len(keyPts)<K:
    '''寻找最大误差可分点'''
    SSE = np.inf
    for i in range(len(keyPts)):
        tempData = train[np.nonzero(Labels==i)[0]]
        KM.train(tempData,2)
        splitDists = KM.dists
        splitSSE = sum(splitDists)
        nonsplitSSE = sum(Dists[np.nonzero(Labels!=i)[0]])
        if splitSSE+nonsplitSSE<SSE:
            SSE=splitSSE+nonsplitSSE
            bestDists = splitDists
            bestLabels = KM.labels
            bestPts = KM.keyPoints
            bestidx = i
    '''替换中心点、距离、标签'''
    idx = np.nonzero(Labels==bestidx)[0]
    n = 0
    for i in idx:
        Dists[i] = bestDists[n]
        if bestLabels[n] == 0:
            Labels[i] = bestidx
        else:
            Labels[i] = len(keyPts)
        n += 1
    keyPts[bestidx] = bestPts[0].tolist()
    keyPts.append(bestPts[1].tolist())



'''画图'''
markers = ['o','^','+','d','D','h']
colors = ['r','y','b','g','b','r']
plt.figure()
x = train[:,0]
y = train[:,1]
for i in set(Labels):
    x1=[]
    y1=[]
    for j in range(len(Labels)):
        if i==Labels[j]:
            x1.append(x[j])
            y1.append(y[j])
    plt.scatter(x1,y1,marker=markers[int(i)],color=colors[int(i)])
plt.scatter(np.array(keyPts)[:,0],np.array(keyPts)[:,1],linewidths=5,color='k')
plt.show()



'''聚类个数评价：轮廓系数'''
LK = []
m = 0
for data in train:
    n=0
    a = 0
    b = dict()
    avalue = 0
    bvalue = 0
    for subdata in train: 
        if m==n:
            n += 1
            continue
        if Labels[m] == Labels[n]:
            a += KM.eDist(data,subdata)
        else:
            if Labels[n] not in b.keys():
                b[Labels[n]] = 0
            b[Labels[n]] += KM.eDist(data,subdata)
        n += 1
    '''a是点到本簇中其他点的平均距离'''
    avalue = (a/(len(np.nonzero(Labels==Labels[m])[0])-1))
    '''b是点到其他簇中其他点的平均距离的最小值'''
    bvalue = np.min([value/len(np.nonzero(Labels==la)[0]) for la,value in b.items()])
    LK.append((bvalue-avalue)/max(bvalue,avalue))
    m += 1
LKratio = np.mean(LK)
'''轮廓系数：0.76066358485307373'''

from sklearn import metrics
metrics.silhouette_score(train, Labels, metric='euclidean')
'''轮廓系数：0.76066358485307373'''




