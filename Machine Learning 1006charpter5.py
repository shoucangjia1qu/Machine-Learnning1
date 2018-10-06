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

###################逐次逼近法/迭代法求解##################

