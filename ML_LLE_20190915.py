# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:24:40 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
import time
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


os.chdir(r"D:\mywork\test")
#数据准备
xs = np.linspace(0, 10, 1000)
zs = np.sin(xs)
ys = np.random.random(1000)
ax = plt.axes(projection='3d')
plt.figure(figsize=(20,10))
ax.scatter(xs=xs[:300], ys=ys[:300], zs=zs[:300])
ax.scatter(xs=xs[300:600], ys=ys[300:600], zs=zs[300:600])
ax.scatter(xs=xs[600:], ys=ys[600:], zs=zs[600:])
plt.show()
x = np.vstack((xs,ys,zs)).T


#sklearn用法
n = 50          #近邻数量
lle = LocallyLinearEmbedding(n_neighbors=n, n_components=2,  method='standard')
lle.fit(x)
tranx = lle.transform(x)
#画图
print(n)
plt.scatter(tranx[:300,0], tranx[:300,1])
plt.scatter(tranx[300:600,0], tranx[300:600,1])
plt.scatter(tranx[600:,0], tranx[600:,1])
plt.show()


#自编用法
m, n = np.shape(x)
#1、计算W
k = 50          #近邻数量
W = np.zeros((m,m))
for i in range(m):
    n_distance = np.zeros((m))
    xi = x[i,:]
    for j in range(m):
        if i==j:
            n_distance[j] = np.inf
        else:
            xj = x[j,:]
            n_distance[j] = np.linalg.norm(xi-xj)
    n_distance_idx = np.argsort(n_distance)                 #求样本间距离
    n_idx = n_distance_idx[:k]                              #选出近邻样本的下标
    xk = x[n_idx,:]                                         #选出近邻样本
    Zi = np.dot((xi-xk),(xi-xk).T)                          #求解Z
    Zin = np.linalg.inv(Zi)                                 #求解Z的逆矩阵
    wi = Zin.sum(axis=1)/Zin.sum()                          #求解w
    W[i,n_idx] = wi                                         #将w赋值回去

#2、计算特征向量求降维后矩阵
I = np.ones((W.shape))
M = np.dot((I-W),(I-W).T)
lamb, vect = np.linalg.eigh(M)  
tranx2 = vect[:,[-1,-2]]         #取最后两维

#3、画图
plt.scatter(tranx2[:300,0], tranx2[:300,1])
plt.scatter(tranx2[300:600,0], tranx2[300:600,1])
plt.scatter(tranx2[600:,0], tranx2[600:,1])
plt.show()





