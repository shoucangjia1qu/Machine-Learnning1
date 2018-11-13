# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:43:08 2018

@author: ZWD
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy, os
from sklearn import preprocessing
from sklearn.decomposition import PCA
from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
from sklearn.cluster import KMeans
from sklearn import metrics

os.chdir(r"D:\mywork\test\ML_CCB")
##############################
#                            #
#       1、自编应用           #
#                            #
##############################

'''1-1定义类：K－means聚类'''
class kmeans(object):
    '''定义属性'''
    def __init__(self):
        self.K = 0          #目标分类个数
        self.keyPoints = 0  #聚类中心点
        self.trainSet = 0   #训练集
        self.labels = []    #分类结果
        self.dists = []     #距离中心点距离
    
    '''定义欧式距离公式'''
    def eDist(self,v1,v2):
        return(np.linalg.norm(v1-v2))
    
    '''定义初始中心点'''
    def randomPoints(self,train,K):
        m,n = train.shape
        keyPoints = np.zeros((K,n))
        for i in range(n):
            maxvalue = np.max(train[:,i])
            minvalue = np.min(train[:,i])
            for j in range(K):
                keyPoints[j,i] = minvalue + np.random.rand()*(maxvalue-minvalue)
        return keyPoints
    
    '''定义k-means聚类函数'''
    def train(self,train,K):
        self.K = K
        self.trainSet = train
        m,n = train.shape
        labelList = np.zeros(m)
        distList = np.zeros(m)
        keyPoints = self.randomPoints(train,K)
        flag = True
        while flag:
            flag = False
            for i in range(m):
                dists = []
                dists = [self.eDist(train[i,:],keyPoints[j,:]) for j in range(K)]
                mindist = min(dists)
                minIdx = dists.index(mindist)
                if labelList[i] != minIdx:
                    flag = True
                labelList[i] = minIdx
                distList[i] = mindist
            '''迭代中心点'''
            for rank in range(K):
                #找到分类标签为rank的数据并重新计算中心点
                newData = train[np.nonzero(labelList==rank)[0]]
                if newData.sum().sum() != 0:
                    keyPoints[rank] = np.mean(newData,axis=0)
        self.keyPoints = keyPoints
        self.labels = labelList
        self.dists = distList



'''1-3初始化并训练'''
KM = kmeans()

'''1-4改进:二分类聚类'''
K=8
point0 = np.mean(train,axis=0)
keyPts = []
keyPts.append(point0.tolist())
Labels = np.zeros(len(train))
Dists = np.zeros(len(train))
for p in range(len(train)):
    Dists[p] = KM.eDist(train[p,:],point0)
#设置初始总误差
while len(keyPts)<K:
    '''寻找最大误差可分点'''
    SSE = np.inf
    for i in range(len(keyPts)):
        tempData = train[np.nonzero(Labels==i)[0]]
        KM.train(tempData,2)
        splitDists = KM.dists
        splitSSE = sum(splitDists)
        nonsplitSSE = sum(Dists[np.nonzero(Labels!=i)[0]])
        if splitSSE+nonsplitSSE<SSE:
            SSE=splitSSE+nonsplitSSE
            bestDists = splitDists
            bestLabels = KM.labels
            bestPts = KM.keyPoints
            bestidx = i
    '''替换中心点、距离、标签'''
    idx = np.nonzero(Labels==bestidx)[0]
    n = 0
    for i in idx:
        Dists[i] = bestDists[n]
        if bestLabels[n] == 0:
            Labels[i] = bestidx
        else:
            Labels[i] = len(keyPts)
        n += 1
    keyPts[bestidx] = bestPts[0].tolist()
    keyPts.append(bestPts[1].tolist())
labels = Labels
labelCount = dict()
for i in set(labels):
    labelCount[i] = len(labels[labels == i])

'''1-5画图'''
markers = ['o','^','+','d','D','h']
colors = ['r','y','b','g','b','r']
plt.figure()
x = train[:,0]
y = train[:,1]
for i in set(Labels):
    x1=[]
    y1=[]
    for j in range(len(Labels)):
        if i==Labels[j]:
            x1.append(x[j])
            y1.append(y[j])
    plt.scatter(x1,y1,marker=markers[int(i)],color=colors[int(i)])
plt.scatter(np.array(keyPts)[:,0],np.array(keyPts)[:,1],linewidths=5,color='k')
plt.show()

'''1-6聚类个数评价：轮廓系数'''
def LK(train,labels):
    LK = []
    m = 0
    for data in train:
        n=0
        a = 0
        b = dict()
        avalue = 0
        bvalue = 0
        for subdata in train: 
            if m==n:
                n += 1
                continue
            if Labels[m] == Labels[n]:
                a += KM.eDist(data,subdata)
            else:
                if Labels[n] not in b.keys():
                    b[Labels[n]] = 0
                b[Labels[n]] += KM.eDist(data,subdata)
            n += 1
        '''a是点到本簇中其他点的平均距离'''
        avalue = (a/(len(np.nonzero(Labels==Labels[m])[0])-1))
        '''b是点到其他簇中其他点的平均距离的最小值'''
        bvalue = np.min([value/len(np.nonzero(Labels==la)[0]) for la,value in b.items()])
        LK.append((bvalue-avalue)/max(bvalue,avalue))
        m += 1
    LKratio = np.mean(LK)
    return(LKratio)

LKratio = LK(train,labels)
'''0.46146660978'''


##############################
#                            #
#     2、数据处理             #
#                            #
##############################

##############################
#                            #
#     3、开始聚类             #
#                            #
##############################
'''3-1用sklearn封装好的算法'''
kmeans = KMeans(n_clusters=)
result=kmeans.fit(train)
labels = result.labels_
'''记录每个标签类别的数量'''
labelCount = dict()
for i in set(labels):
    labelCount[i] = len(labels[labels == i])
'''3-2轮廓系数验证，最终选择8类'''
print(metrics.silhouette_score(train, labels, metric='euclidean'))
##############################
#                            #
#     4、结果解读             #
#                            #
##############################
'''4-1先用决策树拟合'''
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=5, min_samples_leaf=5, random_state=1234) 
clf.fit(train, labels)
'''4-2展示决策树分类效果'''
import pydotplus
from IPython.display import Image
import sklearn.tree as tree
dot_data = tree.export_graphviz(clf, 
                                out_file=None, 
                                feature_names=dataCol,  
                                class_names=['0','1','2','3','4','5','6','7'],
                                filled=True) 
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 


