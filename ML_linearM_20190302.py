# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 22:11:04 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir("D:\\mywork\\test")

#自编数据集
X = np.random.random((200,3))
X[:,1] = X[:,1]+1
X[:,2] = X[:,2]*9
Y = 2.2*X[:,0] + 1.56*X[:,1] + 3*X[:,2] + 2.5
deltaY = (np.random.random(200)-0.5)*0.5
Y = Y + deltaY
#最小二乘法
class linearM(object):
    #1、属性
    def __init__(self):
        self.w = 0      #斜率
        self.b = 0      #截距
        self.sqrLoss = 0    #最小均方误差
        self.trainSet = 0   #训练集
        self.preY = 0   #Y的预测值
    
    #2、最小二乘法训练(只适用于一个变量)
    def trainLm(self,x,Y):
        m,n = np.shape(x)
        #3-1 求斜率
        xmean = np.mean(x,axis=0)
        w = np.dot(Y,(x-xmean))/(np.sum(np.power(x,2),axis=0)-np.power(np.sum(x,axis=0),2)/m)
        #3-2 求截距
        b = np.sum(Y-np.dot(x,w))/m
        #3-3 求预测值和均方误差
        pre = np.dot(x,w) + b
        loss = np.sum((Y-pre)**2)
        self.w = w
        self.b = b
        self.sqrLoss = loss
        self.trainSet = x
        self.preY = pre
    
    #3、矩阵方法求解
    def trainMa(self,x,Y):
        m,n = np.shape(x)
        x2 = np.ones((m,n+1))
        x2[:,:n] = x
        EX = np.linalg.inv(np.dot(x2.T,x2))
        w = np.dot(np.dot(EX,x2.T),Y)
        pre = np.dot(x2,w)
        loss = np.sum((Y-pre)**2)
        self.w = w[:-1]
        self.b = w[-1]
        self.sqrLoss = loss
        self.trainSet = x
        self.preY = pre
    
    #4、梯度下降法求解
    def trainGd(self,x,Y,r,steps):
        m,n = np.shape(x)
        x2 = np.ones((m,n+1))
        x2[:,:n] = x
        w = np.ones(n+1)
        for i in range(steps):
            err = np.dot(x2,w)-Y
            gra = np.dot(err,x2)
            w = w-r*gra/m
            pre = np.dot(x2,w)
            loss = np.sum((Y-pre)**2)
            if loss<0.01:
                break
        self.w = w[:-1]
        self.b = w[-1]
        self.sqrLoss = loss
        self.trainSet = x
        self.preY = pre
        
        
#开始训练
#矩阵求解方法
LM = linearM()
LM.trainMa(X,Y)
w1 = LM.w
b1 = LM.b
preY1 = LM.preY
loss1 = LM.sqrLoss
#梯度下降方法
LM = linearM()
LM.trainGd(X,Y,0.05,5000)
w2 = LM.w
b2 = LM.b
preY2 = LM.preY
loss2 = LM.sqrLoss
#直接求解法好像不适用
LM = linearM()
LM.trainLm(X,Y)
w3 = LM.w
b3 = LM.b
preY3 = LM.preY
loss3 = LM.sqrLoss

#Logit回归
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir("D:\\mywork\\test")
#1、数据集
with open(r"D:\mywork\test\ML\dataSet_BP.txt",'r') as f:
    content=f.readlines()
dataList = [[float(i) for i in row.split()] for row in content]
dataArray = np.array(dataList)
train = dataArray[:,:2]
#调整样本分类
label = np.ones(train.shape[0])
for i in range(307):
    if train[i,1]<train[i,0]*1.28+3:
        label[i] = 0
#label[np.nonzero(train[:,1]<10)[0]] = 0

#2、梯度下降法和牛顿法解决逻辑斯蒂回归最大似然估计问题
class logit(object):
    #属性
    def __init__(self):
        self.w = 0
        self.preY = 0
        self.trainSet = 0
        
    #训练，梯度下降法
    def trainGd(self,X,Y,r,steps):
        m,n = np.shape(X)
        w = np.ones((1,n))
        errorList = []
        for i in range(steps):
            wx = np.dot(X,w.T)
            p = np.exp(wx)/(1+np.exp(wx))
            err = p - Y.reshape(-1,1)
            gra = np.sum(np.multiply(X,err) ,axis=0)
            w = w - r*gra/m
            errorList.append(err.sum())
        self.w = w
        self.preY = p
        self.trainSet = X
        self.errorList = errorList

X = np.hstack(( train, np.ones((train.shape[0],1)) ))
#开始训练：梯度下降法
lg = logit()
lg.trainGd(X,label,0.08,2000)
w1 = lg.w
pre1 = lg.preY
error1 = lg.errorList
#画图1（分类图）
Idx0 = np.nonzero(label==0)[0]
Idx1 = np.nonzero(label==1)[0]
plt.scatter(X[Idx0,0], X[Idx0,1], c='r',marker='^')
plt.scatter(X[Idx1,0], X[Idx1,1], c='b',marker='o')
Xre = np.linspace(-5,15,100)
Yre = -(w1[0,2]+Xre*w1[0,0])/w1[0,1]
plt.plot(Xre,Yre) 
plt.show()
#画图2（误差图）
plt.plot(error1,range(2000))
plt.show()

























