# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:16:56 2019

@author: ZWD
"""

import numpy as np
import pandas as pd
import os, copy
import matplotlib.pyplot as plt


##########线性神经网络，用y=x作为激活函数###########
X = np.ones((200,3))
X[:,:2] = np.random.random((200,2))
X[(2*X[:,0] + 4.4*X[:,1])<3,2] = -1
Y = X[:,2].reshape(-1,1)
X = np.hstack((X[:,:2],np.ones((200,1))))
plt.scatter(X[Y[:,0]==0,0],X[Y[:,0]==0,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.show()

r=0.005
w = np.ones((3,1))
errList=[]
for i in range(1000):
    preY = np.dot(X,w)
    err = Y - preY
    Gra = np.dot(X.T, err)
    w += r*Gra
    errList.append(sum(err**2))

a = -w[0]/w[1]
b = -w[2]/w[1]

testx = np.linspace(-0.5,1.5,20)
testy = a*testx+b

plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.plot(testx, testy, c='black')
plt.show()

plt.plot(range(1000),errList)
plt.show()


##########单层感知器，用y=logit()作为激活函数###########
X = np.ones((200,3))
X[:,:2] = np.random.random((200,2))
X[(2*X[:,0] + 4.4*X[:,1])<3,2] = 0
Y = X[:,2].reshape(-1,1)
X = np.hstack((X[:,:2],np.ones((200,1))))
plt.scatter(X[Y[:,0]==0,0],X[Y[:,0]==0,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.show()


def logit(z):
    return 1/(1+np.exp(-z))

r=0.005
w = np.ones((3,1))
errList=[]
for i in range(2000):
    preY = logit(np.dot(X,w))
    err = Y - preY
    Gra = np.dot(X.T, np.multiply(np.multiply(err,preY),(1-preY)))
    w += r*Gra
    errList.append(sum(err**2))

a = -w[0]/w[1]
b = -w[2]/w[1]

testx = np.linspace(-0.5,1.5,20)
testy = a*testx+b

plt.scatter(X[Y[:,0]==0,0],X[Y[:,0]==0,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.plot(testx, testy, c='black')
plt.show()

plt.plot(range(2000),errList)
plt.show()


##############单层感知器，用y=sign()作为激活函数###########
X = np.ones((200,3))
X[:,:2] = np.random.random((200,2))
X[(2*X[:,0] + 4.4*X[:,1])<3,2] = -1

Y = X[:,2].reshape(-1,1)
X = np.hstack((X[:,:2],np.ones((200,1))))

plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.show()

r=0.01
w = np.ones((3,1))
errList=[]
for i in range(200):
    preY = np.sign(np.dot(X,w))
    err = Y - preY
    Gra = np.dot(X.T, err)
    w += r*Gra
    errList.append(sum(err**2)[0])

a = -w[0]/w[1]
b = -w[2]/w[1]

testx = np.linspace(-0.5,1.5,20)
testy = a*testx+b

plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.plot(testx, testy, c='black')
plt.show()

plt.plot(range(200),errList)
plt.show()


#################统计学习上的感知机，单梯度
X = np.ones((200,3))
X[:,:2] = np.random.random((200,2))
X[(2*X[:,0] + 4.4*X[:,1])<3,2] = -1

Y = X[:,2].reshape(-1,1)
X = np.hstack((X[:,:2],np.ones((200,1))))

plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.show()

r=100       #r可以取很大的数
w = np.ones((3,1))
errList=[]
for i in range(200):
    preY = np.sign(np.dot(X,w))
    err = Y - preY
    for i in range(200):
        if Y[i,0]*(np.dot(X[i,:], w))<0:
            Gra = np.dot(X[i,:].reshape(-1,1), Y[i,0])
            w += r*Gra
    errList.append(sum(err**2)[0])
    
a = -w[0]/w[1]
b = -w[2]/w[1]

testx = np.linspace(-0.5,1.5,20)
testy = a*testx+b

plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.plot(testx, testy, c='black')
plt.show()

plt.plot(range(200),errList)
plt.show()


#################统计学习上的感知机，对偶形式
X = np.ones((200,3))
X[:,:2] = np.random.random((200,2))
X[(2*X[:,0] + 4.4*X[:,1])<3,2] = -1

Y = X[:,2].reshape(-1,1)
X = np.hstack((X[:,:2],np.ones((200,1))))

plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.show()

XXT = np.dot(X,X.T)
alpha = np.zeros((200,1))
#b=0
r = 10000
for i in range(10000):
    num_zero = 0
    for x in range(200):
        if Y[x,0]*np.dot(np.multiply(alpha,Y).T, XXT[:,x].reshape(-1,1)) <= 0:
            print(x)
            print(Y[x,0]*np.dot(np.multiply(alpha,Y).T, XXT[:,x].reshape(-1,1)))
            alpha[x,0] += r
#            b += Y[x,0]
            num_zero += 1
    if num_zero == 0:
        print('迭代%d次'%(i+1))
        break
w = np.multiply(np.multiply(alpha, Y), X).sum(axis=0)

a = -w[0]/w[1]
b = -w[2]/w[1]

testx = np.linspace(-0.5,1.5,20)
testy = a*testx+b

plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r',marker='D')
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='o')
plt.plot(testx, testy, c='black')
plt.show()

