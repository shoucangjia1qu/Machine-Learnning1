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

'''基于用户的推荐(User CF)'''
import numpy as np
import operator

#夹角余弦距离公式
def cosDist(v1,v2):
    dist = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return dist
#KNN算法
def KNN(testData,trainSet,listClass,k):
    x,y = trainSet.shape
    distList = np.zeros(x)      #初始化
    for index in range(x):
        distList[index] = cosDist(testData,trainSet[index])
    indexList = np.argsort(-distList)        #降序排列后的标签列表
    voteDict = dict()       #初始化投票
    for i in range(k):
        voteLabel = listClass[indexList[i]]
        voteDict[voteLabel] = voteDict.get(voteLabel,0) + 1
    sortVote = sorted(voteDict.items(),key=operator.itemgetter(1),reverse=True)     #根据第2个阈值来降序排列
    return sortVote[0][0]

dataMat=np.array([[0.238,0,0.1905,0.1905,0.1905,0.1905],
               [0,0.177,0,0.294,0.235,0.294],
               [0.2,0.16,0.12,0.12,0.2,0.2]])
testSet=[0.2174,0.2174,0.1304,0,0.2174,0.2174]
classLabel=np.array(['B','C','D'])
reClass = KNN(testSet,dataMat,classLabel,3)     #D


'''基于产品的推荐Item CF'''
import numpy as np
import operator

#夹角余弦距离公式
def cosDist(v1,v2):
    dist = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return dist
#KNN算法
def KNN_item(testData,trainSet,labelList,k):
    length = trainSet.shape[0]
    distList = np.zeros(length)     #初始化距离列表
    for index in range(length):
        distList[index] = cosDist(testData,trainSet[index])
    indexList = np.argsort(-distList)
    voteDict = dict()
    for i in range(k):
        voteLabel = labelList[indexList[i]]
        voteDict[voteLabel] = voteDict.get(voteLabel,0) + 1
    sortVote = sorted(voteDict.items(), key=operator.itemgetter(1),reverse=True)
    return sortVote[0][0]

dataSet = np.array([[0.417,0,0.25,0.333],
                    [0.3,0.4,0,0.3],
                    [0,0,0.625,0.375],
                    [0.278,0.222,0.222,0.278],
                    [0.263,0.211,0.263,0.263]
        ])
testData = [0.334,0.333,0,0.333]
labelList = ['B','C','D','E','F']
result = KNN_item(testData,dataSet,labelList,3)



