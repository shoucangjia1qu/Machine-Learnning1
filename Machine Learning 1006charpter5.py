# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:27:40 2018

@author: ecupl
"""

###################逐次逼近法/迭代法求解##################
import numpy as np
import matplotlib.pyplot as plt
'''消元法'''
A = np.mat([[8,-3,2],[4,11,-1],[6,3,12]])
b = np.mat([[20,33,36]])
result = np.linalg.solve(A,b.T)
print(result)
'''迭代法'''
error = 1.0e-6
steps = 100
n=3
B0=np.mat([[0,3/8,-2/8],[-4/11,0,1/11],[-6/12,-3/12,0]])
f=np.mat([[20/8,33/11,36/12]])
xk = np.zeros((n,1))
errorlist=[]
for k in range(steps):
    xk_1 = xk
    xk = B0*xk + f.T
    errorlist.append(np.linalg.norm(xk-xk_1))
    print('第{}次迭代,解是{},误差{}'.format(k+1,xk,errorlist[-1]))
    print('-------------------------------')
    if errorlist[-1] < error:
        break
#画图    
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure()
x = np.linspace(1,k+1,k+1)
y = errorlist
plt.scatter(x,y,c='r',marker='^',linewidths=5)
plt.xlabel('迭代次数')
plt.ylabel('误差')
plt.show()

###################线性感知器##################
x1=np.linspace(0,9,10)
x3=3*x1+2+np.random.rand(10)*0.2
data1=np.array((x1,x3)).T
y1=np.linspace(10,19,10)
y3=3*y1+2-np.random.rand(10)*0.2
data2=np.array((y1,y3)).T
a=data1.tolist()
b=data2.tolist()
a.extend(b)
trainSet = np.array(a)
Bi = np.ones((20,1))
train = np.column_stack((Bi,trainSet))          #生成训练集数据
label = np.row_stack((np.ones((10,1)),-np.ones((10,1))))            #生成训练集标签
'''创建单层感知器'''
X = np.array([[1,3,3],
              [1,4,3],
              [1,1,1]])
#标签
Y = np.array([[1],
              [1],
              [-1]])

W0 = np.zeros((3,1))
alpha = 0.11
steps = 500
for i in range(steps):
    gradient = np.dot(X,W0)     #计算梯度
    O = np.sign(gradient)
    Wt = W0 + alpha*np.dot(X.T,(Y-O))       #计算误差
    W0 = Wt













