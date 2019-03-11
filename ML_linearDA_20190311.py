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

#直接使用葡萄据数据集
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
label = np.array(df.iloc[:,0])
train = np.array(df.iloc[:,1:])

#直接sklearn实现LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()   #选取两个主成分
lda.fit(train,label)
X = lda.transform(train)
lda.explained_variance_ratio_
'''array([0.68747889, 0.31252111])'''
#降维后画图
plt.scatter(xt[:,0],xt[:,1])
plt.show()

#自编算法实现LDA
class Linerda(object):
    #1、设置属性
    def __init__(self):
        self.w = 0
        self.n_components = 0
        self.ratio = 0
        
    #2、数据标准化
    def scale(self,trainSet):
        return (trainSet-trainSet.mean(axis=0))/trainSet.std(axis=0)
    
    #3、进行降维
    def train(self,x,y):
        data = self.scale(x)
        Sb = 0          #类间散度
        Sw = 0          #类内散度
        Miu = np.mean(data,axis=0)       #总体平均数
        ylabel = set(y)
        for i in ylabel:
            Xi = data[np.nonzero(y==i)[0]]
            Miui =  np.mean(Xi,axis=0)
            Swi = np.dot((Xi-Miui).T,(Xi-Miui))
            Sw += Swi
            Sbi = len(Xi)*np.dot((Miui-Miu),(Miui-Miu).T)
            Sb += Sbi
            












