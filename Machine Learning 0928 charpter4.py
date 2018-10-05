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

###################SVD分解高维矩阵##################
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

###################k-means聚类实现##################
import numpy as np
import copy

class Kmeans(object):
    '''初始化'''
    def __init__(self):
        self.CluPoint = 0      #初始化族群的中心点
        self.CluNumber = 0      #初始化族群个数
        self.dataSet = 0        #初始化数据集
        self.labels = 0        #初始化分类标签
        self.dist = 0         #初始化数据点到中心的位置
        
    '''读取数据函数'''
    def loadData(self,path):
        recordList = []
        with open(path,'r') as f:
            content = f.readlines()
        recordList=[row.split() for row in content]
        x,y=np.shape(recordList)
        dataSet=np.zeros((x,y))    #初始化矩阵
        for i in range(x):
            for j in range(y):
                dataSet[i][j] = float(recordList[i][j])
        self.dataSet = dataSet
        
    '''定义欧氏距离公式'''
    def Edist(self,v1,v2):
        return (np.linalg.norm(v1-v2))
    
    '''随机生成初始聚类中心'''
    def randCluPoint(self,dataSet,k):
        x,y = np.shape(dataSet)
        CluPts = np.zeros((k,y))        #初始化随机中心点
        for i in range(y):
            maxv = np.max(dataSet[:,i])
            minv = np.min(dataSet[:,i])
            for j in range(k):
                CluPts[j,i] = minv + np.random.rand()*(maxv-minv)
        return CluPts
    
    '''聚类主函数'''
    def KM(self,dataSet,k):
        self.CluNumber = k
        Label = np.zeros(len(dataSet))      #初始化每个数据的标签
        Dist = np.zeros(len(dataSet))       #初始化每个数据的距离
        centerPoints = self.randCluPoint(dataSet,self.CluNumber)     #随机生成初始中心点
        flag = True
        while flag:
            flag = False
            '''求每个点到每个聚类中心的位置'''
            for i in range(len(dataSet)):
                distList = []
                distList = [self.Edist(dataSet[i,:],centerPoints[j,:]) for j in range(k)]
                minDist = np.min(distList)      #最短距离
                minIndex = distList.index(minDist)      #最短距离的类
                '''判定当前数据点的最小距离和分类是否一致'''
                if Label[i] != minIndex:
                    flag = True
                Label[i] = minIndex
                Dist[i] = minDist
            '''迭代中心位置，每列的平均值'''
            for cent in range(k):
                newData = dataSet[np.nonzero(Label==cent)[0]]       #定位不同类别数据点的下标
                centerPoints[cent,:] = np.mean(newData, axis=0)
        self.labels = Label
        self.dist = Dist
        self.CluPoint = centerPoints

#运行函数
Km = Kmeans()
Km.loadData("D:\\mywork\\test\\ML\\4k2_far_data.txt")
trainData = Km.dataSet[:,1:]
Km.KM(trainData,4)
labels = Km.labels
#画图
import matplotlib.pyplot as plt
x = list(trainData[:,0])
y = list(trainData[:,1])
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

###################二分k-means聚类实现##################
Km=Kmeans()
Km.loadData("D:\\mywork\\test\\ML\\4k2_far_data.txt")
trainData = Km.dataSet[:,1:]
k=4
#初始化中心点
point0 = np.mean(trainData,axis=0)
#聚类中心点集合
cPts = []
cPts.append(point0.tolist())
#计算数据点到初始中心的距离
CluLabel = np.zeros(len(trainData))
CluDist = np.zeros(len(trainData))
for p in range(len(trainData)):
    CluDist[p] = Km.Edist(cPts[0],trainData[p])
#达到规定聚类点数量就停止划分
while (len(cPts)<k):
    SSE = np.inf
    for i in range(len(cPts)):
        '''需要划分的临时数据集'''
        tempData = trainData[np.nonzero(CluLabel==i)[0]]
        '''进行二分类划分'''
        Km.KM(tempData,2)
        tempLabels = Km.labels
        tempDist = Km.dist
        tempCent = Km.CluPoint
        '''划分后数据集的误差'''
        splitSSE = np.sum(tempDist)
        '''不在划分数据集范围内的误差'''
        nonsplitSSE = np.sum(CluDist[np.nonzero(CluLabel != i)[0]])
        '''选取总体误差最小的划分集，也就是寻找最需要划分的数据集'''
        if splitSSE + nonsplitSSE < SSE:
            #重新赋值总体误差，以寻求更小的总体误差
            SSE = splitSSE + nonsplitSSE
            bestCent = tempCent
            bestSplitPts = i
            bestDist = copy.deepcopy(tempDist)
            bestLabels = copy.deepcopy(tempLabels)
    '''替换原先的聚类中心点、标签、距离'''
    #先定义需要替换的下标
    replaceIndex = np.nonzero(CluLabel == bestSplitPts)[0]
    #替换分类标签和最短距离
    for j in range(len(bestLabels)):
        CluDist[replaceIndex[j]] = bestDist[j]
        if bestLabels[j] == 0:
            CluLabel[replaceIndex[j]] = bestSplitPts
        else:
            CluLabel[replaceIndex[j]] = len(cPts)
    #替换聚类中心点
    cPts[bestSplitPts] = bestCent.tolist()[0]
    cPts.append(bestCent.tolist()[1])
        
#画图
import matplotlib.pyplot as plt
x = list(trainData[:,0])
y = list(trainData[:,1])
markers = ['o','^','+','d','h','+']
colors = ['r','y','b','g','g','r']
n = 0
plt.figure()
for label in set(CluLabel):
    x1 = []
    y1 = []
    for i in range(len(CluLabel)):
        if CluLabel[i]==label:
            x1.append(x[i])
            y1.append(y[i])
    plt.scatter(x1,y1,marker=markers[n],color=colors[n])
    n+=1
plt.show()

###################SVD详解##################
'''手工算出U、Sigma、V'''
A=np.mat([[5,5,3,0,5,5],[5,0,4,0,4,4],[0,3,0,5,4,5],[5,4,3,3,5,5]])
lamda,hU = np.linalg.eig(A*A.T)
lamda1,hV = np.linalg.eig(A.T*A)
Sigma = np.sqrt(lamda)
'''numpy算'''
U,S,V=np.linalg.svd(A)
'''SVD简易推荐系统'''
import numpy as np

trainSet=np.array([[0,0,0,0,0,4,0,0,0,0,5],[0,0,0,3,0,4,0,0,0,0,3],
				[0,0,0,0,4,0,0,1,0,4,0],[3,3,4,0,0,0,0,2,2,0,0],
				[5,4,5,0,0,0,0,5,5,0,0],[0,0,0,0,5,0,1,0,0,5,0],
				[4,3,4,0,0,0,0,5,5,0,1],[0,0,0,4,0,4,0,0,0,0,4],
				[0,0,0,2,0,2,5,0,0,1,2],[0,0,0,0,5,0,0,0,0,4,0]])
testSet=np.array([[1,0,0,0,0,0,0,1,2,0,0]])
eps = 1.0e-6

def cosDist(v1,v2):
    Dist=(np.dot(v1,v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)+eps)
    return Dist
U,S,Vt = np.linalg.svd(trainSet)
'''求解奇异值数量
m = 0
for i in S:
    m+=i**2
n=0
for i in S:
    n+=i**2
    if n/m>=0.9:
        break
'''
k=3
Uk = U[:,:k]
Sigma = np.diag(S)[:k,:k]
Vtk = Vt[:k,:]
#得出测试集新坐标
testU = np.dot(np.dot(testSet[0],Vtk.T),np.linalg.inv(Sigma))
#计算所有夹角余弦
Dists = np.array([cosDist(testU,i) for i in Uk])
#从大到小把下标排序
DistsIndex = np.argsort(-Dists)
mostLike = trainSet[DistsIndex[0]]
mostPre = Dists[DistsIndex[0]]
print("相似度最高：{}。推荐{}".format(mostPre,mostLike))



