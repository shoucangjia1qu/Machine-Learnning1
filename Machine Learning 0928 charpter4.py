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

'''SVD计算用户和产品特征'''
import numpy as np

'''定义夹角余弦公式'''
def cosDist(v1,v2):
    dist = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return dist

'''SVD分解高维矩阵'''
data = np.array([
        [5,5,3,0,5,5],
        [5,0,4,0,4,4],
        [0,3,0,5,4,5],
        [5,4,3,3,5,5],
        ])                  #U*I
dataNow = data.T            #I*U
U,Sigma,VT = np.linalg.svd(dataNow)
#Sigma百分比>=90%，故K选2
k=2
#选取前两个奇异值
Sigmak = np.diag(Sigma)[:k,:k]
#训练集产品特征向量
Uk = U[:,:k]
#训练集用户特征向量
Vk = VT[:k,:].T
'''根据矩阵转变特征(U和V是正交矩阵，Sigma是对角矩阵)
    M = U*Sigma*Vt
    Mt = V*Sigma*Ut
    Mt*U*Sigma(-1) = V
    M*V*Sigma(-1) = U'''
testData=np.mat([[5],[5],[0],[0],[0],[5]])          #测试集用户
#测试集用户特征向量
Vtest = testData.T*Uk*np.linalg.inv(Sigmak)
'''或者'''
testData1=np.array([[5],[5],[0],[0],[0],[5]])          #测试集用户
Vtest1 = np.dot(np.dot(testData.T,Uk),np.linalg.inv(Sigmak))
'''夹角余弦计算相似度'''
maxV = 0
maxI = 0
index = 0
for i in Vk:
    tempValue = cosDist(i,np.array(Vtest)[0])
    if tempValue>maxV:
        maxV = tempValue
        maxI = index
    index+=1
print("相似度：{}".format(maxV))
print("和用户{}最相似".format(maxI))

'''直接用U表示用户特征，SVD分解高维矩阵'''
data = np.array([
        [5,5,3,0,5,5],
        [5,0,4,0,4,4],
        [0,3,0,5,4,5],
        [5,4,3,3,5,5],
        ])                  #U*I
U,Sigma,VT = np.linalg.svd(data)
#Sigma百分比>=90%，故K选2
k=2
#选取前两个奇异值
Sigmak = np.diag(Sigma)[:k,:k]
#训练集用户特征向量
Uk = U[:,:k]
#训练集产品特征向量
Vk = VT[:k,:].T
'''根据矩阵转变特征(U和V是正交矩阵，Sigma是对角矩阵)
    M = U*Sigma*Vt
    Mt = V*Sigma*Ut
    Mt*U*Sigma(-1) = V
    M*V*Sigma(-1) = U'''
testData=np.mat([[5,5,0,0,0,5]])          #测试集用户
#测试集用户特征向量
Utest = testData*Vk*np.linalg.inv(Sigmak)
'''或者'''
testData1=np.array([[5],[5],[0],[0],[0],[5]])          #测试集用户
Vtest1 = np.dot(np.dot(testData.T,Uk),np.linalg.inv(Sigmak))
'''夹角余弦计算相似度'''
maxV = 0
maxI = 0
index = 0
for i in Uk:
    tempValue = cosDist(i,np.array(Utest)[0])
    if tempValue>maxV:
        maxV = tempValue
        maxI = index
    index+=1
print("相似度：{}".format(maxV))
print("和用户{}最相似".format(maxI))
###结果一样的

'''下面想讲一下数组、矩阵中np.dot,np.multiply,*的用法'''
import numpy as np
A = np.arange(0,4).reshape(2,2)
B = np.arange(1,5).reshape(2,2)
#np.dot，数组和矩阵都是当作矩阵的乘法来处理，点乘
np.dot(A,B)
np.dot(np.mat(A),np.mat(B))
#np.multiply，数组和矩阵都是对应位置相乘
np.multiply(A,B)
np.multiply(np.mat(A),np.mat(B))
#*，数组是对应位置相乘，矩阵是点乘
A*B
np.mat(A)*np.mat(B)

