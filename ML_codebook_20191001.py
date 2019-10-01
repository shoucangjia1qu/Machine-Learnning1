# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 21:27:58 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
from sklearn import linear_model
os.chdir(r"D:\mywork\test")

with open(r"D:\mywork\test\UCI_data\iris.data") as f:
    data = f.readlines()
trainSet = np.array([row.split(',') for row in data[:-1]])
trainSet = trainSet[:,:-1].astype('float')
trainSet = trainSet - trainSet.mean(axis=0)

#1、得到初始D和X
u, s, v = np.linalg.svd(trainSet)
n_comp = 30                                                     #设置的需要稀疏化的列数
dict_data = u[:, :n_comp]
x = linear_model.orthogonal_mp(dict_data, trainSet)             #Y=DX求出X


#2、外循环，判断是否停止迭代，一是次数达标，二是误差足够小
max_iter = 30
tol = 1.0e-6
for i in range(max_iter):
    e = np.linalg.norm(trainSet - np.dot(dict_data, x))         #误差
    if e<=tol:
        break
    print("第%d次,"%(i+1),"误差：%.6f"%e)
    di, xi = update(trainSet, dict_data, x, n_comp)             #内循环迭代D和X
    dict_data = np.copy(di)
    x = np.copy(xi)
    
    
    
#3、内循环，逐列迭代D和K
def update(trainSet, d, x, n_comp):
    """
    使用KSVD更新字典的过程
    逐列更新字典
    """
    dictionary = np.copy(d)
    xi = np.copy(x)
    for i in range(n_comp):
        index = np.nonzero(xi[i, :])[0]                         #将不为0的部分取出
        if len(index) == 0:
            continue
        dictionary[:, i] = 0                                    #初始化字典当前列
        r = (trainSet - np.dot(dictionary, xi))[:, index]       #计算误差
        # 利用svd的方法，来求解更新字典和稀疏系数矩阵
        u, s, v = np.linalg.svd(r, full_matrices=False)
        # 使用左奇异矩阵的第0列更新字典
        dictionary[:, i] = u[:, 0]
        # 使用第0个奇异值和右奇异矩阵的第0行的乘积更新稀疏系数矩阵
        for j,k in enumerate(index):
            xi[i, k] = s[0] * v[0, j]
    return dictionary, xi






