# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:36:17 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir(r"D:\mywork\test")

with open(r"D:\mywork\test\UCI_data\iris.data") as f:
    data = f.readlines()
trainSet = np.array([row.split(',') for row in data[:-1]])
trainSet = trainSet[:,:-1].astype('float')

#多维缩放聚类
class MultiDimensionalScaling(object):
    #1、类属性
    def __init__(self):
        self.V = 0              #特征向量
        self.A = 0              #特征值
        self.distMatrix = 0     #原数据的距离平方矩阵
        self.innerMatrix = 0    #降维后的内积矩阵
        self.X = 0              #数据集
        self.tranX = 0          #降维后的数据集
        
    #2、计算欧式距离
    def calEdist(self, v1, v2):
        return np.linalg.norm((v1-v2))
    
    #3、计算欧式距离矩阵的平方
    def calDistMatrix(self, X):
        m, d = np.shape(X)
        distMatrix = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i != j:
                    vi = X[i]
                    vj = X[j]
                    distMatrix[i,j] = np.power(self.calEdist(vi, vj), 2)
        return distMatrix
    
    #4、计算内积距离1
    def innerDist1(self, distMatrix, m, j):
        return np.sum(distMatrix[:,j])/(2*m)
        
    #5、计算内积距离2
    def innerDist2(self, distMatrix, m, i):
        return np.sum(distMatrix[i,:])/(2*m)
    
    #6、计算内积距离3
    def innerDist3(self, distMatrix, m):
        return np.sum(distMatrix)/(2*m**2)
    
    #7、计算降维后数据的内积矩阵
    def innerProduct(self, X):
        distMatrix = self.calDistMatrix(X)              #原始数据距离矩阵
        m, d = np.shape(X)
        innerDist3 = self.innerDist3(distMatrix, m)     #计算内积距离3
        innerMatrix = np.zeros((m,m))
        for i in range(m):
            innerDist1 = self.innerDist1(distMatrix, m, i)          #计算内积距离1
            for j in range(m):
                innerDist2 = self.innerDist2(distMatrix, m, j)      #计算内积距离2
                innerMatrix[i,j] = -0.5*distMatrix[i,j] + innerDist1 + innerDist2 - innerDist3
        return innerMatrix, distMatrix
    
    #8、训练
    def train(self, X):
        innerMatrix, distMatrix = self.innerProduct(X)      #计算降维后的矩阵内积
        A, V = np.linalg.eigh(innerMatrix)                  #计算特征值和特征向量
        probA = A/np.sum(A)
        A2 = A[np.nonzero(probA>=0.01)[0]]                  #权重要大于等1%
        Amatrix = np.diag(np.power(A2, 0.5))
        Vmatrix = V[:,np.nonzero(probA>=0.01)[0]]
        tranX = np.dot(Vmatrix, Amatrix)
        self.innerMatrix = innerMatrix
        self.distMatrix = distMatrix
        self.X = X
        self.tranX = tranX
        self.A = A2
        self.V = Vmatrix
        return
    
#计算        
if __name__ == "__main__":
    MDS = MultiDimensionalScaling()
    MDS.train(trainSet)
    
    

