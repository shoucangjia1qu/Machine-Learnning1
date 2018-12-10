# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:04:34 2018

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#######################
#                     #
#     最小二乘法       #
#                     #
#######################
'''读取数据集'''
os.chdir(r"D:\mywork\test\ML")
with open("regdataset.txt","r") as f:
    content = f.readlines()
dataList = [[float(y) for y in x.split()] for x in content]
dataSet = np.array(dataList)

'''计算斜率和截距'''
n = len(dataSet)
x = dataSet[:,0]
y = dataSet[:,1]
xmean = np.mean(x)
ymean = np.mean(y)
a = (np.dot(x,y)-n*xmean*ymean)/(sum(np.power(x,2))-n*xmean**2)
b = ymean - a*xmean
'''计算斜率'''
deltax = x-xmean
deltay = y-ymean
a1 = np.dot(deltax,deltay)/np.sum(np.power(deltax,2))
b1 = ymean - a1*xmean
'''画图展示'''
preY = a1*x+b1
plt.figure()
plt.scatter(x,y)
plt.legend(['plot'],loc=2)
plt.plot(x,preY,c='r',linewidth=3)
plt.xlabel("X")
plt.ylabel("Y")
#plt.legend(['preY'],loc=4)
plt.show()

'''另一种计算方法：利用矩阵求解方式求a和b'''
'''Mx*A=Y,求A'''
'''Mx.T*Mx*A = Mx.T*Y'''
'''A = (Mx.T*Mx).I*Mx.T*Y'''
xma = np.ones((n,2))
xma[:,1] = x
Ex = np.dot(xma.T,xma)
ExI = np.linalg.inv(Ex)
A = np.dot(np.dot(ExI,xma.T),y)
b = A[0]
a = A[1]

'''再另一种计算方法，梯度下降法求解'''
train = np.ones((len(dataList),2))
train[:,1] = dataSet[:,0]
Y = dataSet[:,1].reshape((len(dataList),1))
alpha = 0.001
steps = 500
W = np.ones((2,1))
for step in range(steps):
    d = np.dot(train,W)
    E = Y-d
    W = W + alpha*np.dot(train.T,E)
    plt.figure()
    plt.scatter(x,Y)
    preY = W[0] + W[1]*train[:,1]
    plt.plot(x,preY,c='r',linewidth=3)
    plt.show()

#######################
#                     #
#       RBF网络       #
#     加权线性回归     #
#######################
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
os.chdir(r"D:\mywork\test\ML")
'''1、数据读入'''
with open("nolinear.txt","r") as f:
    content = f.readlines()
dataList = [[float(y) for y in x.split()] for x in content]
dataX = np.array(dataList)[:,:2]
dataY = np.array(dataList)[:,2]

'''2、设置激活函数'''
def fuc(x,k):
    return np.exp(np.sqrt(np.dot(x,x.T))/(-2*k**2))

'''3、训练'''
k=0.02
m,n = np.shape(dataX)
testY = np.zeros(m)
for i in range(len(dataX)):
    '''3-1训练权重W'''
    W = np.eye(m)
    testX = dataX[i,:]
    for j in range(len(dataX)):
        targetX = dataX[j,:]
        deltaX = testX - targetX
        W[j,j] = fuc(deltaX,k)
    '''3-2找出斜率和截距X*A=Y'''
    Ex = np.dot(dataX.T,np.dot(W,dataX))
    '''3-3判断多项式是否为0'''
    if np.linalg.det(Ex) != 0:
        '''A = (Mx.T*Mx).I*Mx.T*Y'''
        A = np.dot(np.dot(np.linalg.inv(Ex),dataX.T),np.dot(W,dataY))
        '''3-4预测Y'''
        preY = np.dot(testX,A)
        testY[i] = preY
'''总误差'''
print(sum(np.power((testY-dataY),2)))

'''4、画图'''
plt.figure()
plt.scatter(dataX[:,1],dataY,linewidths=3)
plt.scatter(dataX[:,1],testY,c='r',linewidths=1)
plt.plot(dataX[:,1],testY,c='g')
plt.show()


#################################
#                               #
#        岭回归                 #
#   多元线性回归中的共线性问题   #
#################################
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
os.chdir(r"D:\mywork\test\ML")
'''1、数据读入'''
with open("ridgedata.txt","r") as f:
    content = f.readlines()
dataList = [[float(y) for y in x.split()] for x in content]
train = np.array(dataList)[:,:3]
label = np.array(dataList)[:,-1]

'''2、标准化数据集'''
def normData(train,label):
    normLabel = label-np.mean(label)
    normTrain = (train - np.mean(train,axis=0))/np.var(train,axis=0)
    return normTrain, normLabel
normX, normY = normData(train,label)

'''3、最小二乘法求解'''
m,n = train.shape
steps=30
Ws = np.zeros((steps,n))
Ks = np.zeros((steps,1))
'''A = (Mx.T*Mx+kI).I*Mx.T*Y'''
for i in range(steps):
    k = float(i)/500
    Ks[i] = k
    Ex = np.dot(normX.T,normX)
    kI = k*np.eye(n)
    if np.linalg.det(Ex+kI) != 0:
        A = np.dot(np.dot(np.linalg.inv(Ex+kI),normX.T),normY)
        Ws[i,:] = A
    else:
        print("This matrix is singular,connot do inverse")

'''4、画岭迹图'''
plt.figure()
plt.plot(Ks,Ws[:,0],c='b')
plt.plot(Ks,Ws[:,1],c='r')
plt.plot(Ks,Ws[:,2],c='g')
plt.annotate("feature[1]",xy = (0,Ws[0,0]),color = 'black')
plt.annotate("feature[2]",xy = (0,Ws[0,1]),color = 'black')
plt.annotate("feature[3]",xy = (0,Ws[0,2]),color = 'black')
plt.show()

'''5、对比标准化后的数据集'''
y = normY
W = Ws[10,:]
preY = np.dot(normX,W)
SSE = np.sqrt(sum(np.power((preY-y),2)))
x=list(range(len(y)))
plt.figure()
plt.plot(x,y,c='b')
plt.plot(x,preY,c='r',linewidth=3)
plt.show()
'''SSE=10.396695072100146'''


'''6、还原原始数据集'''
m,n = train.shape
Ex = np.dot(train.T,train)
kI = 0.02*np.eye(n)
if np.linalg.det(Ex+kI) != 0:
    W = np.dot(np.dot(np.linalg.inv(Ex+kI),train.T),label)
else:
    print("This matrix is singular,connot do inverse")
y = label
preY = np.dot(train,W)
SSE = np.sqrt(sum(np.power((preY-y),2)))
x=list(range(len(y)))
plt.figure()
plt.plot(x,y,c='b')
plt.plot(x,preY,c='r',linewidth=3)
plt.show()
'''SSE=10.337942223356293'''

#################################
#                               #
#        Logistic吸引子         #
#                               #
#################################
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
os.chdir(r"D:\mywork\test\ML")
'''1、画图'''
def draw(x1,x2,k):
    plt.figure()
    plt.plot(x1)
    plt.plot(x2)
    plt.title("k=%s"%k)
    plt.show()
'''2、logistic映射迭代函数'''
def logistic_map(k,init):
    maxIters = 50
    x = list(range(maxIters))
    x[0] = init
    for i in range(maxIters-1):
        x[i+1] = k*x[i]*(1-x[i])
    return x
'''k=0.1'''
x1 = logistic_map(0.1,0.1)
x2 = logistic_map(0.1,0.9)
draw(x1,x2,0.1)
'''k=0.9'''
x1 = logistic_map(0.9,0.1)
x2 = logistic_map(0.9,0.9)
draw(x1,x2,0.9)
'''k=1.2'''
x1 = logistic_map(1.2,0.1)
x2 = logistic_map(1.2,0.9)
draw(x1,x2,1.2)
'''k=2.8'''
x1 = logistic_map(2.8,0.1)
x2 = logistic_map(2.8,0.9)
draw(x1,x2,2.8)
'''k=3'''
x1 = logistic_map(3,0.1)
x2 = logistic_map(3,0.9)
draw(x1,x2,3)
'''k=3.5'''
x1 = logistic_map(3.5,0.1)
x2 = logistic_map(3.5,0.9)
draw(x1,x2,3.5)
'''k=3.6'''
x1 = logistic_map(3.6,0.1)
x2 = logistic_map(3.6,0.9)
draw(x1,x2,3.6)
'''k=3.8'''
x1 = logistic_map(3.8,0.1)
x2 = logistic_map(3.8,0.9)
draw(x1,x2,3.8)
'''k=4'''
x1 = logistic_map(4,0.1)
x2 = logistic_map(4,0.9)
draw(x1,x2,4)

'''绘制k[2,4]的值'''
maxIters = 1000
k = np.linspace(2.1,4.0,maxIters)
klen = len(k)
xMat = np.zeros((klen,maxIters))
x = 1/float(maxIters)       #初始值
for i in range(klen):
    for j in range(maxIters):
        x = float(k[i]*x*(1-x))     #指定k后进行迭代
        xMat[i,j] = x
'''画图'''
plt.figure()
for i in range(klen):
    plt.scatter(k,xMat[:,i],s=0.1,marker='.')
plt.show()


