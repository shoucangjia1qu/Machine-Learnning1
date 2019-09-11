# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:18:43 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

os.chdir(r"D:\mywork\test")

with open(r"D:\mywork\test\UCI_data\iris.data") as f:
    data = f.readlines()
trainSet = np.array([row.split(',') for row in data[:-1]])
trainSet = trainSet[:,:-1].astype('float')
x = (trainSet-trainSet.mean(axis=0))/trainSet.std(axis=0)


#sklearn方法
Kpca = KernelPCA(kernel="rbf", n_components=2,  fit_inverse_transform=True, gamma=0.03)
Kpca.fit(x)
ks = Kpca.fit_transform(x)           #结果
alpha = Kpca.alphas_                        #核矩阵的特征向量
lam = Kpca.lambdas_                         #核矩阵的特征值
#画图
plt.scatter(ks[:50,0], ks[:50,1])
plt.scatter(ks[50:100,0], ks[50:100,1])
plt.scatter(ks[100:,0], ks[100:,1])
plt.show()

#自编的方法
##1、求核转换矩阵
m, n =x.shape
Kma = np.zeros((m,m))
gamma = 0.03
for i in range(m):
    vi = x[i]
    for j in range(m):
        vj = x[j]
        Kma[i,j] = np.exp((-1)*gamma*(np.power(np.linalg.norm(vi-vj),2)))
#中心化核矩阵
l = np.ones((m,m))
kc=Kma-np.dot(l,Kma)/m-np.dot(Kma,l)/m+np.dot(np.dot(l,Kma),l)/(m*m)
lamd, lamvect = np.linalg.eigh(kc)
lamd = lamd[::-1][:2]
lamvect = lamvect[:,[-1,-2]]
ks2 = np.dot(kc, lamvect)
#除以特征值的模的根号
for idx, row in enumerate(lamd):
    ks2[:,idx] = ks2[:,idx]/(row**0.5)
#画图
plt.scatter(ks2[:50,0], ks2[:50,1])
plt.scatter(ks2[50:100,0], ks2[50:100,1])
plt.scatter(ks2[100:,0], ks2[100:,1])
plt.show()





