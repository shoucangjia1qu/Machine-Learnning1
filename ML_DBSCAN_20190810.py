# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:39:23 2019

@author: ecupl
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import copy

os.chdir(r"D:\mywork\test")


#创建DBSCAN类
class DBSCAN(object):
    #1、类的属性
    def __init__(self):
        self.dataSet = 0            #数据集
        self.labels = 0             #聚类标签
        self.Clusters = 0           #簇的样本
        self.ClusterDict = {}       #簇聚类的字典
        self.Unvisits = 0           #噪音点
        self.n_clusters = 0         #聚类数
        self.cntPoints = 0          #聚类中心点
        self.r = 0                  #邻域半径
        self.MinPts = 0             #邻域内最小样本数
        
    #2、计算距离公式
    def calDist(self, v1, v2):
        """
        input:
            向量v1、v2
        return:
            向量间的欧式距离
        """
        return np.linalg.norm((v1-v2))
    
    #3、计算生成密度直达邻域的函数
    def calCircle(self, r, center, x, dataIdx):
        """
        input:
            r:邻域半径
            center:中心点
            x:数据集
            dataIdx:数据下标
        return:
            n:邻域内密度直达个数
            nebxIdx:领域内样本的下标
        """
        nebIdx = []
        for i in dataIdx:
            v = x[i]
            dist = self.calDist(center, v)
            if dist <= r:
                nebIdx.append(i)
        n = len(nebIdx)
        return n, nebIdx
           
    #4、从一个集合中删除另一个集合中的元素
    def delset(self, ParentIdx, ChildIdx):
        """
        input:
            ParentIdx:母集，需要删除
            ChildIdx:子集，删除母集中的子集
        return:
            newParentIdx:删除子集内元素后的母集
            DeleteIdx:从母集中删除的子集集合
        """
        newParentIdx = copy.deepcopy(ParentIdx)
        DeleteIdx = []
        for idx in ChildIdx:
            if idx in ParentIdx:
                newParentIdx.remove(idx)
                DeleteIdx.append(idx)
            else:
                pass
        return newParentIdx, DeleteIdx
    
    #5、内循环，判断是否继续要进行迭代
    def innerLoop(self, x, r, MinPts, centerIdx, UnvisitIdx):
        m, d = np.shape(x)
        dataIdx = list(range(m))
        center = x[centerIdx]
        n, nebIdx = self.calCircle(r, center, x, dataIdx)
        if n >= MinPts:
            for subcenterIdx in nebIdx:
                if subcenterIdx not in UnvisitIdx:
                    continue
                elif subcenterIdx in self.Clusters:
                    continue
                self.Clusters.append(subcenterIdx)
                if subcenterIdx == centerIdx:
                    continue
                else:
                    self.innerLoop(x, r, MinPts, subcenterIdx, UnvisitIdx)
                    self.Clusters.append(subcenterIdx)
        else:
            return
        return
            
    #6、训练
    def train(self, x, r, MinPts):
        """
        input:
            x:数据集
            r:邻域半径
            MinPts:邻域内最小个数
        return:
            labels:簇标签
            centerPoints:中心点
        """
        self.dataSet = x
        self.r = r
        self.MinPts = MinPts
        m, d = np.shape(x)
        centerPoints = []           #存放核心点的列表
        labels = np.zeros(m)        #簇标签列表
        dataIdx = list(range(m))    #数据下标
        k = 0                       #初始簇
        ##1、寻找核心点
        for idx in dataIdx:
            center = x[idx]         #邻域中心点
            n, nebIdx = self.calCircle(r, center, x, dataIdx)           #计算中心点邻域密度直达的样本个数和下标
            if n >= MinPts:
                centerPoints.append(idx)                                #对于大于最小领域内个数的样本下标加入到中心点
        UnvisitIdx = list(range(m)) #未访问的样本下标
        ##2、根据核心点，寻找密度可达的点
        while len(centerPoints) != 0:
            centerIdx = np.random.choice(centerPoints)
            self.Clusters = []  #初始化簇的下标
            self.innerLoop(x, r, MinPts, centerIdx, UnvisitIdx)     #内循环递归生成簇
            ClusterList = list(set(self.Clusters))                  #簇内元素去重
            newUnvisitIdx, DeleteUnvisitIdx = self.delset(UnvisitIdx, ClusterList)          #未访问的样本下标去除已生成的簇中样本下标
            newCenterPoints, DeleteCenterIdx = self.delset(centerPoints, ClusterList)       #中心点样本去除已生成的簇中样本下标
            UnvisitIdx = copy.deepcopy(newUnvisitIdx)
            centerPoints = copy.deepcopy(newCenterPoints)
            self.ClusterDict[centerIdx] = ClusterList               #簇的字典{中心点：簇中其他点}
            labels[ClusterList] = k                                 #标签
            k += 1
        ##3、得到最终的聚类结果
        self.Unvisits = UnvisitIdx
        self.n_clusters = k
        self.labels = labels
        self.cntPoints = x[list(self.ClusterDict.keys())]
        return

#正式训练
if __name__ == "__main__":
    data = np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],[0.403,0.237],[0.481,0.149],
                 [0.437,0.211],[0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],
                 [0.360,0.370],[0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],[0.748,0.232],
                 [0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],[0.751,0.489],[0.532,0.472],[0.473,0.376],
                 [0.725,0.445],[0.446,0.459]])
    db = DBSCAN()
    db.train(data, 0.11, 5)
    ClusterDict = db.ClusterDict
    Unvisit = db.Unvisits
    labels = db.labels
    for centerIdx, nebIdx in ClusterDict.items():
        plt.scatter(data[nebIdx,0], data[nebIdx,1])
        #核心点
        plt.scatter(data[centerIdx,0], data[centerIdx,1], c='r', marker='s',linewidths=5)
    #噪音点
    plt.scatter(data[Unvisit,0], data[Unvisit,1], c='r', marker='*',linewidths=5)
    plt.show()
        
#评价指标        
##轮廓系数
def LK(train,Labels):
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
                a += db.calDist(data,subdata)
            else:
                if Labels[n] not in b.keys():
                    b[Labels[n]] = 0
                b[Labels[n]] += db.calDist(data,subdata)
            n += 1
        '''a是点到本簇中其他点的平均距离'''
        avalue = (a/(len(np.nonzero(Labels==Labels[m])[0])-1))
        '''b是点到其他簇中其他点的平均距离的最小值'''
        bvalue = np.min([value/len(np.nonzero(Labels==la)[0]) for la,value in b.items()])
        LK.append((bvalue-avalue)/max(bvalue,avalue))
        m += 1
    LKratio = np.mean(LK)
    return(LKratio)
print(LK(data,labels))
       
##DB指数
###1、欧式距离
def calEdist(v1, v2):
    return np.linalg.norm((v1-v2))

###2、簇内平均距离
def avgCluster(x):
    dist = 0
    m, d = np.shape(x)
    for i in range(m):
        for j in range(m):
            if j > i:
                dist += calEdist(x[i], x[j])
    return dist/(m*(m-1))

###3、簇内最大距离
def maxCluster(x):
    maxdist = 0
    m, d = np.shape(x)
    for i in range(m):
        for j in range(m):
            if j > i:
                dist = calEdist(x[i], x[j])
            else:
                continue
            if dist > maxdist:
                maxdist = copy.deepcopy(dist)
    return maxdist

###4、两簇间中心点距离
def centerClusters(x1, x2):
    m1, d1 = np.shape(x1)
    m2, d2 = np.shape(x2)
    miu1 = np.sum(x1, axis=0)/m1
    miu2 = np.sum(x2, axis=0)/m2
    return calEdist(miu1, miu2)

###5、两簇间最近样本距离
def minClusters(x1, x2):
    mindist = np.inf
    m1, d1 = np.shape(x1)
    m2, d2 = np.shape(x2)
    for i in range(m1):
        for j in range(m2):
            dist = calEdist(x1[i], x2[j])
            if dist < mindist:
                mindist = copy.deepcopy(dist)
    return mindist

###6、DB指数
def calDBI(train, label):
    labelSet = np.unique(label)
    k = len(labelSet)
    DBIList = []
    for i in labelSet:
        maxDBI = 0
        for j in labelSet:
            if i==j:
                continue
            xi = train[label==i]
            xj = train[label==j]
            DBI = (avgCluster(xi) + avgCluster(xj))/centerClusters(xi, xj)
            if DBI > maxDBI:
                maxDBI = copy.deepcopy(DBI)
        DBIList.append(maxDBI)
    return sum(DBIList)/k, DBIList

print(calDBI(data, labels))

##Dunn指数
def calDI(train, label):
    labelSet = np.unique(label)
    k = len(labelSet)
    clusterMax = 0              #计算簇内样本间最大距离
    for i in labelSet:
        xl = train[label==i]
        clusterDist = maxCluster(xl)
        if clusterDist > clusterMax:
            clusterMax = copy.deepcopy(clusterDist)
    minOutCluster = []
    for i in labelSet:
        minInCluster = np.inf
        for j in labelSet:
            if i==j:
                continue
            xi = train[label==i]
            xj = train[label==j] 
            Incluster = minClusters(xi, xj)/clusterMax
            if Incluster < minInCluster:
                minInCluster = copy.deepcopy(Incluster)
        minOutCluster.append(minInCluster)
    return min(minOutCluster), minOutCluster
            
print(calDI(data, labels))















