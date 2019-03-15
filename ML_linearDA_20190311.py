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
lda.fit(data,label)
X = lda.transform(data)
lda.explained_variance_ratio_
'''array([0.68747889, 0.31252111])'''
#降维后画图
plt.scatter(X[:,0],X[:,1])
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
            Swi = np.dot((Xi-Miui).T,(Xi-Miui))/(len(Xi)-1)
            Sw += Swi
            Sbi = len(Xi)*np.dot((Miui-Miu).reshape(1,-1).T,(Miui-Miu).reshape(1,-1))
            Sb += Sbi
        U,S,V = np.linalg.svd(Sw)
        #选取2个变量
        w = np.dot(Sb,np.dot(np.dot(V.T[:,:2],np.diag(S[:2])),U.T[:2,:2]))

X=np.dot(data,w)

plt.scatter(X[label==1,0],X[label==1,1],c='r')
plt.scatter(X[label==2,0],X[label==2,1],c='b')
plt.scatter(X[label==3,0],X[label==3,1],c='y')
plt.show()



######西瓜书3.5课后习题
dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
    ]
#特征值列表
labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
#整理出数据集和标签
X = np.array(dataSet)[:,6:8]
X=X.astype(float)
Y = np.array(dataSet)[:,8]
Y[Y=="好瓜"]=1
Y[Y=="坏瓜"]=0
Y=Y.astype(float)

#二分类问题
#1、计算类内散度
def calSw(X,Y):
    Sw = 0
    Miui = []
    Swi = []
    for i in set(Y):
        Xi = X[Y==i,:]
        u = np.mean(Xi,axis=0)      #计算第i类的均值
        Miui.append(u)
        s = np.dot((Xi-u).T,(Xi-u)) #计算第i类的协方差
        Swi.append(s)
        Sw += s
    return Sw,Miui,Swi

Sw,Miui,Swi = calSw(X,Y)
#2、计算类间散度
Sb = Miui[0] - Miui[1]
#3、计算W向量
U,S,V = np.linalg.svd(Sw)
Sn = np.linalg.inv(np.diag(S))
Swn = np.dot(np.dot(V.T,Sn),U.T)
w = np.dot(Swn,Sb.reshape(-1,1))
#4、求投影后的距离
Xw = np.dot(X,w)
X2 = np.dot(X,w)/np.linalg.norm(w)
#5、求投影后的点
a = w[1]/w[0]           #向量的斜率
Xnew = np.zeros((17,2))     #投影后的点
for i in range(17):
    x0=np.sqrt(X2[i,0]**2/(1+a**2))
    y0=a*x0
    Xnew[i,0]=x0
    Xnew[i,1]=y0
#或者另一种算法#
Xnew2 = np.zeros((17,2))
for i in range(17):
    x0 = np.dot(X[i,:],w)*w[0]/np.power(np.linalg.norm(w),2)
    y0 = x0*a
    Xnew2[i,0]=x0
    Xnew2[i,1]=y0

#6、画图
plt.figure(figsize=(8,6))
plt.scatter(X[Y==1,0],X[Y==1,1],c='g',marker='+')
plt.scatter(X[Y==0,0],X[Y==0,1],c='r',marker='d')
plt.plot([0,0.15],[0,0.15*a])
plt.scatter(Xnew[Y==1,0],Xnew[Y==1,1],c='g',marker='o')
plt.scatter(Xnew[Y==0,0],Xnew[Y==0,1],c='r',marker='o')
for i in range(17):
    if Y[i]==1:
        plt.plot([X[i,0],Xnew[i,0]],[X[i,1],Xnew[i,1]],'g--')
    else:
        plt.plot([X[i,0],Xnew[i,0]],[X[i,1],Xnew[i,1]],'r--')
plt.show()

######如果直接用sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()   #选取两个主成分
lda.fit(X,Y)
X3 = lda.transform(X)
lda.explained_variance_ratio_
#预测和评价
preY = lda.predict(X)
from sklearn import metrics
print(metrics.classification_report(Y,preY))

#降维后画图
plt.scatter(X[:,0],X[:,1])
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(X[Y==1,0],X[Y==1,1],c='g',marker='+')
plt.scatter(X[Y==0,0],X[Y==0,1],c='r',marker='d')
plt.plot([0,0.15],[0,0.15*a])
plt.scatter(Xnew[preY==1,0],Xnew[preY==1,1],c='g',marker='o')
plt.scatter(Xnew[preY==0,0],Xnew[preY==0,1],c='r',marker='o')
for i in range(17):
    if Y[i]==1:
        plt.plot([X[i,0],Xnew[i,0]],[X[i,1],Xnew[i,1]],'g--')
    else:
        plt.plot([X[i,0],Xnew[i,0]],[X[i,1],Xnew[i,1]],'r--')
plt.scatter([Xnew[Y==1,0].mean(),Xnew[Y==0,0].mean()],[Xnew[Y==1,1].mean(),Xnew[Y==0,1].mean()],c='y',marker='D',linewidths=10)
plt.show()



