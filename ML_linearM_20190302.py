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

#自编数据集
X = np.random.random((200,3))
X[:,1] = X[:,1]+1
X[:,2] = X[:,2]*9
Y = 2.2*X[:,0] + 1.56*X[:,1] + 3*X[:,2] + 2.5
deltaY = (np.random.random(200)-0.5)*0.5
Y = Y + deltaY
#最小二乘法
class linearM(object):
    #1、属性
    def __init__(self):
        self.w = 0      #斜率
        self.b = 0      #截距
        self.sqrLoss = 0    #最小均方误差
        self.trainSet = 0   #训练集
        self.preY = 0   #Y的预测值
    
    #2、最小二乘法训练(只适用于一个变量)
    def trainLm(self,x,Y):
        m,n = np.shape(x)
        #3-1 求斜率
        xmean = np.mean(x,axis=0)
        w = np.dot(Y,(x-xmean))/(np.sum(np.power(x,2),axis=0)-np.power(np.sum(x,axis=0),2)/m)
        #3-2 求截距
        b = np.sum(Y-np.dot(x,w))/m
        #3-3 求预测值和均方误差
        pre = np.dot(x,w) + b
        loss = np.sum((Y-pre)**2)
        self.w = w
        self.b = b
        self.sqrLoss = loss
        self.trainSet = x
        self.preY = pre
    
    #3、矩阵方法求解
    def trainMa(self,x,Y):
        m,n = np.shape(x)
        x2 = np.ones((m,n+1))
        x2[:,:n] = x
        EX = np.linalg.inv(np.dot(x2.T,x2))
        w = np.dot(np.dot(EX,x2.T),Y)
        pre = np.dot(x2,w)
        loss = np.sum((Y-pre)**2)
        self.w = w[:-1]
        self.b = w[-1]
        self.sqrLoss = loss
        self.trainSet = x
        self.preY = pre
    
    #4、梯度下降法求解
    def trainGd(self,x,Y,r,steps):
        m,n = np.shape(x)
        x2 = np.ones((m,n+1))
        x2[:,:n] = x
        w = np.ones(n+1)
        for i in range(steps):
            err = np.dot(x2,w)-Y
            gra = np.dot(err,x2)
            w = w-r*gra/m
            pre = np.dot(x2,w)
            loss = np.sum((Y-pre)**2)
            if loss<0.01:
                break
        self.w = w[:-1]
        self.b = w[-1]
        self.sqrLoss = loss
        self.trainSet = x
        self.preY = pre
        
        
#开始训练
#矩阵求解方法
LM = linearM()
LM.trainMa(X,Y)
w1 = LM.w
b1 = LM.b
preY1 = LM.preY
loss1 = LM.sqrLoss
#梯度下降方法
LM = linearM()
LM.trainGd(X,Y,0.05,5000)
w2 = LM.w
b2 = LM.b
preY2 = LM.preY
loss2 = LM.sqrLoss
#直接求解法好像不适用
LM = linearM()
LM.trainLm(X,Y)
w3 = LM.w
b3 = LM.b
preY3 = LM.preY
loss3 = LM.sqrLoss



###################Logit回归######################
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir("D:\\mywork\\test")
#1、数据集
with open(r"D:\mywork\test\ML\dataSet_BP.txt",'r') as f:
    content=f.readlines()
dataList = [[float(i) for i in row.split()] for row in content]
dataArray = np.array(dataList)
train = dataArray[:,:2]
#调整样本分类
label = np.ones(train.shape[0])
for i in range(307):
    if train[i,1]<train[i,0]*1.28+3:
        label[i] = 0
#label[np.nonzero(train[:,1]<10)[0]] = 0

#2、梯度下降法和牛顿法解决逻辑斯蒂回归最大似然估计问题
class logitall(object):
    #属性
    def __init__(self):
        self.w = 0
        self.preY = 0
        self.trainSet = 0
        
    #训练，梯度下降法
    def trainGd(self,X,Y,r,steps):
        m,n = np.shape(X)
        w = np.ones((1,n))
        errorList = []
        for i in range(steps):
            wx = np.dot(X,w.T)
            p = np.exp(wx)/(1+np.exp(wx))
            err = p - Y.reshape(-1,1)
            gra = np.sum(np.multiply(X,err) ,axis=0)
            w = w - r*gra/m
            errorList.append(err.sum())
        self.gra = gra
        self.w = w
        self.preY = p
        self.trainSet = X
        self.errorList = errorList
    
    #训练，牛顿法
    def trainNt(self,X,Y,steps):
        m,n = np.shape(X)
        w = np.ones((1,n))
        errorList = []
        for i in range(steps):
            wx = np.dot(X,w.T)
            p = np.exp(wx)/(1+np.exp(wx))
            err = p - Y.reshape(-1,1)
            gra = np.sum(np.multiply(X,err) ,axis=0)
            Hx = np.zeros((n,n))
            for j in range(m):
                xxt = np.dot(X[j,:].reshape(-1,1),X[j,:].reshape(1,-1))
                Hx += xxt*p[j]*(1-p[j])
#            Hx = np.dot(np.dot(np.dot(X.T,np.diag(p.reshape(m))),np.diag(1-p.reshape(m))),X)
            w = w - np.dot(gra,np.linalg.inv(Hx)/m)
            errorList.append(err.sum())
        self.w = w
        self.preY = p
        self.trainSet = X
        self.errorList = errorList
    
    def calGra(self, X, Y, w):
        wx = np.dot(X,w.T)
        p = np.exp(wx)/(1+np.exp(wx))
        err = p - Y.reshape(-1,1)
        gra = np.sum(np.multiply(X,err) ,axis=0)
        return gra.reshape(1,-1)
    
    #训练：拟牛顿法：DFP算法
    def trainDFP(self, X, Y, r, steps=10, e=1.0e-4, N=10):
        m, n = np.shape(X)
        Dk0 = np.eye(n)             #初始的Dk0矩阵
        w0 = np.ones((1,n))         #初始的权重矩阵
        g0 = self.calGra(X,Y,w0)    #计算初始的梯度
        if np.linalg.norm(g0)<e:
            self.w = w0
            return
        for i in range(steps):
            Pk = -np.dot(g0, Dk0)   #计算下降方向
            minFx = np.inf
            w1 = np.zeros((1,n))
            bestr = 0
            for n in range(N):      #一维搜素最优步长，并进行迭代
                wi = w0 + (r**n)*Pk
                Fx = sum(-np.multiply(Y.reshape(-1,1),np.dot(X,wi.T))+\
                         np.log(1+np.exp(np.dot(X,wi.T))))      #求函数最小值
                if Fx < minFx:
                    minFx = Fx
                    w1 = np.copy(wi)
                    bestr = r**n
            print('函数最小值:',minFx,'最好的步长：',bestr)
            g1 = self.calGra(X,Y,w1) #计算迭代后的梯度
            print('梯度：',np.linalg.norm(g1))
            if np.linalg.norm(g1)<e:
                self.w = w1
                break
            deltaW = w1 - w0
            deltaG = g1 - g0
            #计算新的Dk1矩阵
            Dk1 = Dk0 + np.dot(deltaW.T, deltaW)/np.dot(deltaW, deltaG.T) - \
            np.dot(np.dot(np.dot(Dk0, deltaG.T), deltaG), Dk0)/np.dot(np.dot(deltaG, Dk0), deltaG.T)
            #赋予新的值
            Dk0 = np.copy(Dk1)
            w0 = np.copy(w1)
            g0 = np.copy(g1)
        if isinstance(self.w, int):
            self.w = w1
        wx = np.dot(X,self.w.T)
        p = np.exp(wx)/(1+np.exp(wx))
        self.iters = i
        self.preY = p
        self.trainSet = X
        self.gra = g1 if g1 is not None else g0
    
    #训练：拟牛顿法：BFGS算法
    def trainBFGS(self, X, Y, r, steps=5, e=1.0e-4, N=10):
        m, n = np.shape(X)
        Bk0 = np.eye(n)             #初始的Bk0矩阵
        w0 = np.ones((1,n))         #初始的权重矩阵
        g0 = self.calGra(X,Y,w0)    #计算初始的梯度
        if np.linalg.norm(g0)<e:
            self.w = w0
            return
        for i in range(steps):
            Bk0ni = np.linalg.inv(Bk0)  #求Bk0的逆矩阵
            Pk = -np.dot(g0, Bk0ni)   #计算下降方向
            minFx = np.inf
            w1 = np.zeros((1,n))
            bestr = 0
            for n in range(N):      #一维搜素最优步长，并进行迭代
                wi = w0 + (r**n*Pk)
                Fx = sum(-np.multiply(Y.reshape(-1,1),np.dot(X,wi.T))+\
                         np.log(1+np.exp(np.dot(X,wi.T))))      #求函数最小值
                if Fx < minFx:
                    minFx = Fx
                    w1 = np.copy(wi)
                    bestr = r**n
            print('函数最小值:',minFx,'最好的步长：',bestr,'最优次数：',n)
            g1 = self.calGra(X,Y,w1) #计算迭代后的梯度
            print('梯度的模：',np.linalg.norm(g1))
            if np.linalg.norm(g1)<e:
                self.w = w1
                break
            deltaW = w1 - w0
            deltaG = g1 - g0
            #计算新的Bk1矩阵
            P = np.dot(deltaG.T, deltaG)/np.dot(deltaW, deltaG.T)
            Q = np.dot(np.dot(np.dot(Bk0, deltaW.T), deltaW), Bk0)/np.dot(np.dot(deltaW, Bk0), deltaW.T)
            Bk1 = Bk0 + P - Q
            #赋予新的值
            Bk0 = np.copy(Bk1)
            w0 = np.copy(w1)
            g0 = np.copy(g1)
        if isinstance(self.w, int):
            self.w = w1
        wx = np.dot(X,self.w.T)
        p = np.exp(wx)/(1+np.exp(wx))
        self.iters = i
        self.preY = p
        self.trainSet = X
        self.gra = g1 if g1 is not None else g0

    #训练：拟牛顿法：LBFGS算法
    def trainLBFGS(self, X, Y, r, steps=5, e=1.0e-4, N=10, alpha=0.5):
        m, n = np.shape(X)
        Gk0 = np.eye(n)             #初始的Gk0矩阵
        w0 = np.ones((1,n))         #初始的权重矩阵
        g0 = self.calGra(X,Y,w0)    #计算初始的梯度
        if np.linalg.norm(g0)<e:
            self.w = w0
            return
        for i in range(steps):
            Pk = -np.dot(g0, Gk0)   #计算下降方向
            minFx = np.inf
            w1 = np.zeros((1,n))
            bestr = 0
            for ni in range(N):      #一维搜素最优步长，并进行迭代
                wi = w0 + (r**ni*Pk)
                Fx = sum(-np.multiply(Y.reshape(-1,1),np.dot(X,wi.T))+\
                         np.log(1+np.exp(np.dot(X,wi.T))))      #求函数最小值
                if Fx < minFx:
                    minFx = Fx
                    w1 = np.copy(wi)
                    bestr = r**ni
            print('函数最小值:',minFx,'最好的步长：',bestr,'最优次数：',n)
            g1 = self.calGra(X,Y,w1) #计算迭代后的梯度
            print('梯度的模：',np.linalg.norm(g1))
            if np.linalg.norm(g1)<e:
                self.w = w1
                break
            deltaW = w1 - w0
            deltaG = g1 - g0
            G_DFP = Gk0 + np.dot(deltaW.T, deltaW)/np.dot(deltaW, deltaG.T) - \
            np.dot(np.dot(np.dot(Gk0, deltaG.T), deltaG), Gk0)/np.dot(np.dot(deltaG, Gk0), deltaG.T)
            G_BFGS =np.dot(np.dot((np.eye(n)-np.dot(deltaW.T,deltaG)/np.dot(deltaW,deltaG.T)), Gk0),\
            np.eye(n)-np.dot(deltaG.T,deltaW)/np.dot(deltaW,deltaG.T)) + np.dot(deltaW.T,deltaW)/np.dot(deltaW,deltaG.T)
            Gk1 = alpha*G_DFP + (1-alpha)*G_BFGS
            #赋予新的值
            Gk0 = np.copy(Gk1)
            w0 = np.copy(w1)
            g0 = np.copy(g1)
        if isinstance(self.w, int):
            self.w = w1
        wx = np.dot(X,self.w.T)
        p = np.exp(wx)/(1+np.exp(wx))
        self.iters = i
        self.preY = p
        self.trainSet = X
        self.gra = g1 if g1 is not None else g0








X = np.hstack(( train, np.ones((train.shape[0],1)) ))
#开始训练：梯度下降法
lg = logit()
lg.trainGd(X,label,0.08,2000)
w1 = lg.w
pre1 = lg.preY
error1 = lg.errorList

#开始训练：牛顿法
lg = logit()
lg.trainNt(X,label,2000)
w2 = lg.w
pre2 = lg.preY
error2 = lg.errorList

#画图1（分类图）
Idx0 = np.nonzero(label==0)[0]
Idx1 = np.nonzero(label==1)[0]
plt.scatter(X[Idx0,0], X[Idx0,1], c='r',marker='^')
plt.scatter(X[Idx1,0], X[Idx1,1], c='b',marker='o')
Xre = np.linspace(-5,15,100)
Yre = -(w1[0,2]+Xre*w1[0,0])/w1[0,1]
plt.plot(Xre,Yre) 
plt.show()
#画图2（误差图）
plt.plot(range(2000),error1)
plt.show()







######西瓜书课后练习
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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
X = np.array(dataSet)
X=X.astype(float)
Y = np.array(dataSet)[:,8]
Y[Y=="好瓜"]=1
Y[Y=="坏瓜"]=0
Y=Y.astype(int)

#对数几率分类器
class logit(object):
    #属性
    def __init__(self):
        self.w = 0
        self.label = 0
        self.trainSet = 0
        self.errorList = 0
    
    #方法：对数几率预测函数
    def prelogit(self,wx):
        return 1/(1+np.exp(-wx))

    #方法：牛顿法迭代参数
    def trainNw(self,X,Y,steps):
        errorList = []
        Y2 = Y.reshape(-1,1)
        m,n = np.shape(X)
        X2 = np.hstack((X,np.ones((m,1))))
        w = np.ones((1,n+1))
        for i in range(steps):
            wx = np.dot(X2,w.T)         #内积
            Gx = np.dot((self.prelogit(wx)-Y2).T,X2)        #梯度
            Hx = np.dot(np.multiply(X2,self.prelogit(wx)*(1-self.prelogit(wx))).T,X2)       #二阶梯度
            w -= np.dot(Gx,np.linalg.inv(Hx))        #迭代参数
            error = np.sum(self.prelogit(wx)-Y2)        #错误率
            errorList.append(error)
            if abs(error) < 0.000001:           #判断是否退出
                break
        self.w = w
        self.trainSet = X
        self.label = Y
        self.errorList = errorList



#用牛顿法训练      
lg = logit()
lg.trainNw(Xdata,Ylabel,10)
error = lg.errorList
w = lg.w
#画图
a=-w[0,0]/w[0,1]    #斜率
b=-w[0,2]/w[0,1]     #截距
plt.scatter(X[Y==1,0],X[Y==1,1],c='b',marker='+')
plt.scatter(X[Y==0,0],X[Y==0,1],c='r',marker='d')
plt.plot(np.linspace(0,0.8,10),a*np.linspace(0,0.8,10)+b)
plt.show()

m,n = np.shape(X[:,1].reshape(-1,1))
X2 = np.hstack((X[:,1].reshape(-1,1),np.ones((m,1))))
wx = np.dot(X2,w.T)
P = lg.prelogit(wx)

#换成梯度下降试试
m,n = np.shape(X2)
w = np.ones((1,3))
errorList = []
for i in range(5000):
    wx = np.dot(X2,w.T)
    p = np.exp(wx)/(1+np.exp(wx))
    err = p - Y.reshape(-1,1)
    gra = np.sum(np.multiply(X2,err) ,axis=0)
    w = w - 0.8*gra/m
    errorList.append(err.sum())

#预测和评价
preLogit = lg.prelogit(np.dot(X2,w.T))
preY = np.copy(preLogit)
preY[preY>=0.5]=1
preY[preY<0.5]=0
from sklearn import metrics
print(metrics.classification_report(Y,preY))








#############用全部的特征进行分类##############

#用sklearn试试
import sklearn.linear_model as linear_model
logistic_model = linear_model.LogisticRegression()
logistic_model.fit(Xdata, Ylabel)
logistic_model.predict(Xdata)
logistic_model.coef_

#用statsmodels试试(加上了C()，相当于将变量进行独热编码后进行训练)
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
X3 = np.hstack((Xdata,Ylabel.reshape(-1,1)))
X3 = pd.DataFrame(X3, columns=['s1','s2','s3','s4','s5','s6','s7','s8','re'])
lg = smf.glm('''re ~ C(s1)+s7+s8''', data=X3, 
             family=sm.families.Binomial(sm.families.links.logit)).fit()
lg.summary()

#用拟牛顿法-DFP试试
lg = logitall()
lg.trainDFP(Xdata2[:,5:], Ylabel, 0.6, steps=500)
lg.w
lg.iters

#用拟牛顿法-BFGS试试
lg = logitall()
lg.trainBFGS(Xdata2, Ylabel, 0.6, steps=500)
lg.w
lg.iters

#多分类变量变成哑变量试试
from sklearn.preprocessing import OneHotEncoder
onehotencode = OneHotEncoder(categories='auto',handle_unknown='ignore')
onehotencode.fit(Xdata2[:,0].reshape(-1,1))
onehotdata=onehotencode.transform(Xdata2[:,0].reshape(-1,1)).toarray()    #得到独热编码的数据
Xdata3 = np.hstack((onehotdata[:,1:],Xdata2[:,6:]))

lg = logitall()
lg.trainBFGS(Xdata3, Ylabel, 0.6, steps=500)
lg.w
lg.iters

#用拟牛顿法-LBFGS试试
lg = logitall()
lg.trainLBFGS(Xdata3, Ylabel, 0.6, steps=500)
lg.w
lg.iters



################################
# 用10折交叉验证和留一法进行验证 #
################################
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
Y = np.array(df.iloc[:,0])
X = np.array(df.iloc[:,1:])
Xscale = (X-np.mean(X,axis=0))/np.std(X,axis=0)

lg = logit()
lg.trainNw(Xscale,Y,10)


