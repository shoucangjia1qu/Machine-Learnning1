# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:55:08 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(r"D:\mywork\test")

#准备数据
X = np.linspace(-3,3 , 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, X.shape)
Y = np.square(X) + noise
X = (X-X.mean(axis=0))/X.std(axis=0)

############梯度下降法求解RBF参数##############
#计算径向基距离函数
def guass(alpha, X, Ci):
    return np.exp(-np.linalg.norm((X-Ci), axis=1)**2/(2*alpha**2))
#初始化参数
def init(X, hidden):
    m,n = np.shape(X)
    alpha = np.random.random((hidden,1))        #(h,1)
    C = np.random.random((hidden,n))            #(h,n)
    w = np.random.random((hidden+1,1))          #(h+1,1)
    return alpha, C, w
#将原始数据经过高斯转换
def change(alpha, X, C):
    m,n = np.shape(X)
    newX = np.zeros((m, len(C)))
    for i in range(len(C)):
        newX[:,i] = guass(alpha[i], X, C[i])
    return newX
#给输入层加一列
def AddCol(X):
    return np.hstack((X,np.ones((X.shape[0],1))))
#计算整体误差
def calSSE(Y, preY):
    return 0.5*(np.linalg.norm(Y-preY))**2
#L2范数
def l2(X,C):
    m,n = np.shape(X)
    newX = np.zeros((m, len(C)))
    for i in range(len(C)):
        newX[:,i] = np.linalg.norm((X-C[i]), axis=1)**2
    return newX

#训练
m, n = np.shape(X)
r = 0.000015                          #学习率
errList = []
alpha, C, w = init(X,30)            #初始化参数
for i in range(20000):
    ##正向传播
    hi_output = change(alpha,X,C)       #隐含层输出(m,h)，即通过径向基函数的转换
    yi_input = AddCol(hi_output)        #输出层输入(m,h+1)，因为是线性加权，故将偏置加入
    yi_output = np.dot(yi_input, w)     #输出预测值(m,1)
    errList.append(calSSE(Y,yi_output)) #保存误差
    ##误差反向传播
    deltaw = - np.dot(yi_input.T, (Y-yi_output))    #(h+1,m)x(m,1)
    w -= r*deltaw
    deltaalpha = - np.divide(np.multiply(np.dot(np.multiply(hi_output,l2(X,C)).T, \
                (Y-yi_output)), w[:-1]), alpha**3)  #(h,m)x(m,1)
    alpha -= r*deltaalpha
    deltaC1 = - np.divide(w[:-1],alpha**2)          #(h,1)
    deltaC2 = np.zeros((1,n))                       #(1,n)
    for j in range(m):
        deltaC2 += (Y-yi_output)[j]*np.dot(hi_output[j], X[j]-C)
    deltaC = np.dot(deltaC1,deltaC2)                #(h,1)x(1,n)
    C -= r*deltaC
    if i%100 == 0:
        print(errList[-1])
        #画图
        plt.scatter(X,Y)
        plt.plot(X,yi_output,c='r')
        plt.show()

plt.plot(range(20000), errList)
plt.show()


############完全内插法:解方程法求解RBF参数，固定alpha和中心点##############
X = np.linspace(-3,3 , 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, X.shape)
Y = np.square(X) + noise
X = (X-X.mean(axis=0))/X.std(axis=0)

m, n = np.shape(X)
C = X
alpha = 0.02
G = np.zeros((m,m))
for i in range(m):
    G[:,i] = np.exp(-np.linalg.norm((X-C[i]), axis=1)**2/(2*alpha**2))
w = np.dot(np.linalg.inv(G), Y)

plt.scatter(X,Y)
plt.plot(X,np.dot(G,w),c='r')
plt.show()


############选取几个中心点作为隐含层求解RBF参数##############
X = np.linspace(-3,3 , 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, X.shape)
Y = np.square(X) + noise
X = (X-X.mean(axis=0))/X.std(axis=0)

m, n = np.shape(X)
h = 20
rand = np.random.randint(0,m,h)
C = X[rand,:]
Cdist = np.zeros((h,h))
for  i in range(h):
    for j in range(h):
        if i != j:
            Cdist[i,j] = np.linalg.norm(C[i]-C[j])
alpha = Cdist.max()/(2*h)**0.5
G = np.zeros((m,h))
for i in range(h):
    G[:,i] = np.exp(-np.linalg.norm((X-C[i]), axis=1)**2/(2*alpha**2))
GGT = np.dot(G.T,G)
GGT_inv = np.linalg.inv(GGT)
w = np.dot(np.dot(GGT_inv, G.T), Y)

plt.scatter(X,Y)
plt.plot(X,np.dot(G,w),c='r')
plt.show()

Xi = np.array([15])
Gi = np.exp(-np.linalg.norm((Xi-C), axis=1)**2/(2*alpha**2))
Yi = np.dot(Gi,w.reshape(-1))

plt.scatter(X,Y)
plt.plot(X,np.dot(G,w),c='r')
plt.scatter(Xi,Yi,c='g',linewidths=5)
plt.show()


###############测试,Hermit多项式###################
X = np.linspace(-5,5 , 500)[:, np.newaxis]
Y = np.multiply(1.1*(1-X+2*X**2),np.exp(-0.5*X**2))
plt.scatter(X,Y)
plt.show()













