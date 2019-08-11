# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:51:49 2019

@author: ZWD
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
os.chdir(r"D:\mywork\test")

#K-means聚类
class kmeans(object):
    #1、属性
    def __init__(self):
        self.K = 0          #聚类簇的个数
        self.CntPoint = 0   #聚类中心点
        self.labels = 0     #聚类类别
        self.dists = 0      #每个点距离中心点的距离
    
    #2、距离公式(采用欧式距离)
    def calDist(self,v0,v1):
        return np.linalg.norm(v0-v1)
    
    #3、中心标准化原数据
    def stdData(self,train):
        stdtrain = (train-np.mean(train,axis=0))/np.std(train,axis=0)
        return stdtrain
    
    #4、初始化中心点
    def firstCntPoint(self,train,K):
        m,n=np.shape(train)
        ratio = np.random.random((K,n))
        minvalue = np.min(train,axis=0)
        maxvalue = np.max(train,axis=0)
        points = minvalue+ratio*(maxvalue-minvalue)
        return points
    
    #5、主循环
    def cluster(self,train,K,std=False):
        #4-1 判断数据是否需要标准化
        if std==True:
            train = self.stdData(train)
        #4-2 初始化中心点
        CntPoint = self.firstCntPoint(train,K)
        #4－3 开始聚类
        m,n = np.shape(train)
        labellist = np.zeros(m)
        distlist = np.zeros(m)
        flag = True
        while flag:
            flag = False
            #每个点的聚类标签循环一遍
            for i in range(m):
                dists = [self.calDist(train[i,:],CntPoint[c,:]) for c in range(K)]
                mindist = min(dists)
                minIdx = dists.index(mindist)
                #只要有一个样本的类不一致，就要再次循环
                if minIdx != labellist[i]:
                    flag = True
                labellist[i] = minIdx
                distlist[i] = mindist
            #迭代每个中心点
            for j in range(K):
                dataSet = train[np.nonzero(labellist==j)[0],:]
                if len(dataSet) != 0:
                    CntPoint[j,:] = np.mean(dataSet,axis=0)
                else:
                    print("第{}个点无近似样本".format(j))
        self.K = K
        self.CntPoint = CntPoint
        self.labels = labellist
        self.dists = distlist

#自己造一组数据
train = np.zeros((200,2))
train[:40,:] = np.random.random((40,2))
train[40:102,:] = np.random.random((62,2))*2+2
train[102:150,:] = np.random.random((48,2))*2-3
train[150:,:] = np.random.random((50,2))*5+5
plt.scatter(train[:,0],train[:,1])
plt.show()

#初始化，开始聚类
km = kmeans()
km.cluster(train,4)
km.K
km.CntPoint
km.labels
km.dists

la=km.labels
x=train[:,0]
y=train[:,1]
plt.scatter(x[np.nonzero(la==0)[0]], y[np.nonzero(la==0)[0]],c='b')
plt.scatter(x[np.nonzero(la==1)[0]], y[np.nonzero(la==1)[0]],c='y')
plt.scatter(x[np.nonzero(la==2)[0]], y[np.nonzero(la==2)[0]],c='r')
plt.scatter(x[np.nonzero(la==3)[0]], y[np.nonzero(la==3)[0]],c='g')
plt.show()

'''非常容易陷入局部最优，后面尝试用二分类K-means聚类法'''

#二分类K-means聚类
class kmeans(object):
    #1、属性
    def __init__(self):
        self.K = 0          #聚类簇的个数
        self.CntPoint = []  #聚类中心点
        self.labels = 0     #聚类类别
        self.dists = 0      #每个点距离中心点的距离
    
    #2、距离公式(采用欧式距离)
    def calDist(self,v0,v1):
        return np.linalg.norm(v0-v1)
    
    #3、中心标准化原数据
    def stdData(self,train):
        stdtrain = (train-np.mean(train,axis=0))/np.std(train,axis=0)
        return stdtrain
    
    #4、初始化中心点
    def firstCntPoint(self,train,K):
        m,n=np.shape(train)
        ratio = np.random.random((K,n))
        minvalue = np.min(train,axis=0)
        maxvalue = np.max(train,axis=0)
        points = minvalue+ratio*(maxvalue-minvalue)
        return points
    
    #5、聚类内循环
    def cluster(self,train,K):
        m,n = np.shape(train)
        labellist = np.zeros(m)
        distlist = np.zeros(m)
        #5-1 初始化中心点
        CntPoint = self.firstCntPoint(train,K)
        #5-2 开始聚类
        flag = True
        while flag:
            flag = False
            #每个点的聚类标签循环一遍
            for i in range(m):
                dists = [self.calDist(train[i,:],CntPoint[c,:]) for c in range(K)]
                mindist = min(dists)
                minIdx = dists.index(mindist)
                #只要有一个样本的类不一致，就要再次循环
                if minIdx != labellist[i]:
                    flag = True
                labellist[i] = minIdx
                distlist[i] = mindist
            #迭代每个中心点
            for j in range(K):
                dataSet = train[np.nonzero(labellist==j)[0],:]
                if len(dataSet) != 0:
                    CntPoint[j,:] = np.mean(dataSet,axis=0)
                else:
                    print("第{}个点无近似样本".format(j))
        return CntPoint,labellist,distlist
    
    #6、二分类外循环
    def train(self,trainSet,K,std=False):
        if std==True:
            trainSet = self.stdData(trainSet)
        m,n = np.shape(trainSet)
        labellist = np.zeros(m)                     #初始化聚类标签
        distlist = np.zeros(m)                      #初始化到中心点距离
        pt0 = np.mean(trainSet,axis=0)
        self.CntPoint.append(pt0)                   #初始化中心点
        while len(self.CntPoint)<K:
            SSE = np.inf                #初始化总误差
            for i in range(len(self.CntPoint)):
                Idx = np.nonzero(labellist==i)[0]       #选择当前标签下的数据集下标
                dataSet = trainSet[Idx,:]
                subPoint,sublabel,subdist = self.cluster(dataSet,2)     #进行二分类聚类的内循环
                splitSSE = np.sum(subdist)              #划分过的距离和
                nonsplitSSE = np.sum(distlist[np.nonzero(labellist!=i)[0]])     #未经划分的距离和
                if splitSSE+nonsplitSSE<SSE:
                    SSE = splitSSE+nonsplitSSE
                    bestIdx = i
                    bestCntPoint = copy.deepcopy(subPoint)
                    bestlabel = copy.deepcopy(sublabel)
                    bestdist = copy.deepcopy(subdist)
            replaceIdx = np.nonzero(labellist==bestIdx)[0]          #需要替换的数据下标
            distlist[replaceIdx] = bestdist                         #更新离中心的距离
            for x in range(len(bestlabel)):
                if bestlabel[x] ==0:
                    labellist[replaceIdx[x]] = bestIdx              #二分类结果为0的，标签保留原数
                else:
                    labellist[replaceIdx[x]] = len(self.CntPoint)   #二分类结果为1的，标签储存为新的数
            self.CntPoint[bestIdx] = bestCntPoint[0,:]              #更新原下标位置的中心点
            self.CntPoint.append(bestCntPoint[1,:])                 #加入新下标位置的中心点
        self.K = K
        self.labels = labellist
        self.dists = distlist

#实例化
km = kmeans()
km.train(data,4,std=True)
km.K
km.CntPoint
km.labels
km.dists

la=km.labels
x=data[:,0]
y=data[:,1]
plt.scatter(x[np.nonzero(la==0)[0]], y[np.nonzero(la==0)[0]],c='b')
plt.scatter(x[np.nonzero(la==1)[0]], y[np.nonzero(la==1)[0]],c='y')
plt.scatter(x[np.nonzero(la==2)[0]], y[np.nonzero(la==2)[0]],c='r')
plt.scatter(x[np.nonzero(la==3)[0]], y[np.nonzero(la==3)[0]],c='g')
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
print(LK(data,la))

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

print(calDBI(data, la))

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
            
print(calDI(data, la))














