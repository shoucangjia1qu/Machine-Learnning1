# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:23:44 2019

@author: ZWD
"""

import numpy as np
import pandas as pd
import os, copy
import matplotlib.pyplot as plt

os.chdir(r"D:\mywork\test")

#########SOM网络，自组织映射网络##########
class Kohonen(object):
    def __init__(self):
        self.maxRate = 0.5
        self.minRate = 0.01
        self.maxRound = 5
        self.minRound = 0.1
        self.RateList = []
        self.RoundList = []
        self.dataSet = 0
        self.Labels = 0
        self.steps = 1000
        self.w = 0
        self.Gdist = 0
        self.grid = 0
        
    def normlize(self,X):
        return (X-X.mean(axis=0))/(X.std(axis=0)+1.0e-10)
    
    def edist(self,X1,X2):
        return (np.linalg.norm(X1-X2))
    
    #初始化竞争层
    def init_grid(self,M,N):
        grid = np.zeros((M*N,2))      #分成M*N类，两个维度
        k = 0
        for i in range(M):
            for j in range(N):
                grid[k,:] = np.array([i,j])
                k += 1
        return grid
    
    #学习率和影响半径逐步减少
    def changeRate(self,i):
        Rate = self.maxRate - (self.maxRate-self.minRate)*(i+1)/self.steps
        Round = self.maxRound - (self.maxRound-self.minRound)*(i+1)/self.steps
        return Rate, Round
    
    #计算各个节点之间的距离
    def calGdist(self, grid):
        m = len(grid)
        Gdist = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i != j:
                    Gdist[i,j] = self.edist(grid[i], grid[j])
        return Gdist
    
    #开始训练
    def train(self,X,M,N):
        self.dataSet = X
        ##标准化数据集
        X = self.normlize(X)
        Xrow, Xcol = X.shape
        ##初始化各个节点位置，以及各节点之间的位置
        grid = self.init_grid(M,N)
        self.grid = grid
        self.Gdist = self.calGdist(grid)
        ##初始化各个节点对应的权值
        w = np.random.random((M*N,Xcol))
        ##确定迭代次数并开始迭代，不小于样本数的500倍
        if self.steps<5*Xrow:
            self.steps = 5*Xrow
        for i in range(self.steps):
            ##生成学习率和节点影响半径，并保存
            Rate, Round = self.changeRate(i)
            self.RateList.append(Rate)
            self.RoundList.append(Round)
            ##随机选取样本计算距离
            data = X[np.random.randint(0,Xrow,1)[0],:]
            Xdist = [self.edist(data,w[row]) for row in range(len(w))]
            ##找到优胜节点
            min_gridIdx = Xdist.index(min(Xdist))
            ##确定优胜节点附近需要一起迭代的节点
            RoundIdx = list(np.nonzero(self.Gdist[min_gridIdx]<Round)[0])
            ##对节点权值进行调整
            for wIdx in RoundIdx:
                w[wIdx] += Rate*(data-w[wIdx])
        self.w = w
        ##分类
        self.Labels = np.zeros(Xrow)
        for xIdx in range(Xrow):
            Xi_dist = [self.edist(X[xIdx], w[wIdx]) for wIdx in range(len(w))]
            self.Labels[xIdx] = np.argmin(Xi_dist)

#运行测试
with open(r'D:\mywork\test\UCI_data\iris.data',"r") as f:
    content = f.readlines()
data = np.array([row.split(',') for row in content[:-1]])
X = data[:,:-1].astype('float')
Y = data[:,-1]
for i,j in zip(range(len(set(Y))),set(Y)):
    Y[Y==j] = i
Y = Y.astype('int')

som = Kohonen()
som.train(X,10,10)
grid = som.grid


plt.figure(figsize=(10,8))
for i in set(som.Labels):
    x = X[som.Labels==i,0]
    y = X[som.Labels==i,1]
    plt.scatter(x,y)
    plt.annotate(sum(som.Labels==i), (np.mean(x),np.mean(y)))
plt.show()
#画出拓朴结构
for i in set(som.Labels):
    plt.scatter(grid[int(i),0], grid[int(i),1], linewidths=sum(som.Labels==i)/2)
    plt.annotate(sum(som.Labels==i), (grid[int(i),0], grid[int(i),1]))
plt.xlim((-0.5,10.5))
plt.ylim((-0.5,10.5))
plt.show()


for j in range(150):
    i = som.Labels[j]
    if j<=49:
        plt.scatter(grid[int(i),0], grid[int(i),1], linewidths=sum(som.Labels==i)/2, c='r')
    elif j<=99:
        plt.scatter(grid[int(i),0], grid[int(i),1], linewidths=sum(som.Labels==i)/2, c='b')
    else:
        plt.scatter(grid[int(i),0], grid[int(i),1], linewidths=sum(som.Labels==i)/2, c='y')
    plt.annotate(sum(som.Labels==i), (grid[int(i),0], grid[int(i),1]))
plt.xlim((-0.5,10.5))
plt.ylim((-0.5,10.5))
plt.show()






