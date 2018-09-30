# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:27:40 2018

@author: ecupl
"""

###################推荐系统##################
'''调用k-means聚类算法'''
import numpy as np
#from Recommand_Lib import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#导入数据，转换成矩阵
def file2matrix(path):
    datalist=[]     #初始化列表
    with open(path,"r") as file:
        content=file.readlines()
    datalist=[row.split() for row in content]
    x,y=np.shape(datalist)
    datamatrix=np.zeros([400,3])    #初始化矩阵
    for i in range(x):
        for j in range(y):
            datamatrix[i][j] = float(datalist[i][j])
    return datamatrix

dataSet = file2matrix("D:\\mywork\\test\\ML\\4k2_far_data.txt")
trainSet = dataSet[:,1:]
#执行k-means算法
kmeans =KMeans(n_clusters=4)
kmeans.fit(trainSet)
labels = list(kmeans.labels_)         #生成标签
#画图
x = list(trainSet[:,0])
y = list(trainSet[:,1])
markers = ['o','^','+','d']
colors = ['r','y','b','g']
n = 0
plt.figure()
for label in set(labels):
    x1 = []
    y1 = []
    for i in range(len(labels)):
        if labels[i]==label:
            x1.append(x[i])
            y1.append(y[i])
    plt.scatter(x1,y1,marker=markers[n],color=colors[n])
    n+=1
plt.show()

    

    