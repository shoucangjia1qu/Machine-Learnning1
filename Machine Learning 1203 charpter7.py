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


#######################
#                     #
#       RBF网络       #
#                     #
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

































