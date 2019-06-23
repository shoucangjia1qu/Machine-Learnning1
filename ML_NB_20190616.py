# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:26:37 2019

@author: ecupl
"""

import os
import numpy as np
import pandas as np

os.chdir("D:\\mywork\\test")

######一、朴素贝叶斯分类器
class NBayes(object):
    #设置属性
    def __init__(self):
        self.Y = 0          #训练集标签
        self.X = 0          #训练集数据
        self.PyArr = {}     #先验概率总容器
        self.PxyArr = {}    #条件概率总容器
        
    #连续变量处理，返回均值和标准差
    def Gaussian(self, xArr):
        miu = np.mean(xArr)            #变量平均数
        sigma = np.std(xArr)           #变量标准差
        return miu, sigma

    #离散变量直接计算概率
    def classify(self, x, xArr, countSetX):
        countX = len(xArr)              #计算变量X的数量
        countXi = sum(xArr == x)   #计算变量X某个属性的数量
        Pxy = (countXi+1)/(countX+countSetX)    #加入拉普拉斯修正的概率
        return Pxy
    
    #计算P(y)，加入了拉普拉斯修正
    def calPy(self,Y):
        Py = {}
        countY = len(Y)
        for i in set(Y.flatten()):
            countI = sum(Y[:,0] == i)
            Py[i] = (countI + 1) / (countY + len(set(Y.flatten())))
        self.PyArr = Py
        return
    
    #计算P(x|y)，加入了拉普拉斯修正
    def calPxy(self, X, Y):
        m, n = np.shape(X)
        Pxy = {}
        for yi in set(Y.flatten()):
            countYi = sum(Y[:,0] == yi)
            Pxy[yi] = {}                        #第一层是标签Y的分类
            for xIdx in range(n):
                Pxy[yi][xIdx] = {}              #第二层是不同的变量X
                setX = set(X[:,xIdx])
                tempX = X[np.nonzero(Y[:,0] == yi)[0],xIdx]
                for xi in setX:
                    countSetX = len(setX)
                    if countSetX <= 10:
                        Pxy[yi][xIdx][xi] = self.classify(xi, tempX, countSetX)     #第三层是变量Xi的分类概率，离散变量
                    else:
                        Pxy[yi][xIdx]['miu'], Pxy[yi][xIdx]['sigma'] = self.Gaussian(tempX)
        self.PxyArr = Pxy
        return
    
    #训练
    def train(self, X, Y):
        self.calPy(Y)
        print('P(y)训练完毕')
        self.calPxy(X, Y)
        print('P(x|y)训练完毕')
        self.X = X
        self.Y = Y
        return
    
    #连续变量求概率密度
    def calContinous(self, x, miu, sigma):
        Pxy = np.exp(-(x-miu)**2/(2*sigma**2))/(np.power(2*np.pi,0.5)*sigma)   #计算概率密度
        return Pxy
    
    #预测
    def predict(self, testX):
        preP = {}
        m, n = testX.shape
        for yi, Py in self.PyArr.items():
            Ptest = Py
            print(yi,Ptest)
            for xIdx in range(n):
                xi = testX[0,xIdx]
                if len(set(self.X[:,xIdx])) <= 10:
                    Ptest *= self.PxyArr[yi][xIdx][xi]
                else:
                    pxy = self.calContinous(xi, self.PxyArr[yi][xIdx]['miu'], self.PxyArr[yi][xIdx]['sigma'])
                    Ptest *= pxy
                print(yi,Ptest)
            preP[yi] = Ptest
        return preP
    
    #防止数值下溢，预测时用log
    def predictlog(self, testX):
        preP = {}
        m, n = testX.shape
        for yi, Py in self.PyArr.items():
            Ptest = 0
            for xIdx in range(n):
                xi = testX[0,xIdx]
                if len(set(self.X[:,xIdx])) <= 10:
                    Ptest += self.PxyArr[yi][xIdx][xi]
                else:
                    pxy = self.calContinous(xi, self.PxyArr[yi][xIdx]['miu'], self.PxyArr[yi][xIdx]['sigma'])
                    Ptest += pxy
                print(yi,Ptest)
            preP[yi] = Py*Ptest
        return preP
    
#数据集准备
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

#训练
NB = NBayes()
NB.train(X, Y)
Pdict = NB.predict(X[0,:].reshape(1,-1))
logPdict = NB.predictlog(X[0,:].reshape(1,-1))
                
            



