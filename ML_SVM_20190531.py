# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:01:48 2019

@author: ZWD
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir(r"D:\mywork\test")

#用SMO算法求得最小值
class SMO(object):
    #1、类的属性
    def __init__(self):
        self.Xdata = 0              #测试集数据
        self.Ylabel = 0             #数据实际标签
        self.alpha = 0              #参数
        self.C = 0                  #惩罚因子
        self.Kvalue = dict()        #核函数类型和参数
        self.K = 0                  #核技巧之后的结果
        self.maxiters = 0           #最大循环次数
        self.tol = 0                #容错率
        self.b = 0                  #截距
        self.m = 0                  #样本数量
        self.svindex = 0            #支持向量的下标
        self.SVMvects = 0           #支持向量
        self.SVMlabels = 0          #支持向量的标签
        self.EiMark = 0             #误差是否更新的标识
    
    #2、核函数    
    def kernel(self,X,Z):
        if list(self.Kvalue.keys())[0] == 'Linear':
            Ki = np.dot(X1, Z.T)
        elif list(self.Kvalue.keys())[0] == 'Gaussian':
            L2 = np.power(np.linalg.norm((X-Z), axis=1),2)
            #np.power((X-Z),2).sum(axis=1)
            Ki = np.exp(L2/(-1*self.Kvalue['Gaussian']**2))
        else:
            raise NameError('无法识别的核函数')
        return Ki
    
    
    #3、计算分离超平面的距离
    def distHyperplane(self,i):
        return np.dot(self.K[i,:], np.multiply(self.alpha, self.Ylabel)) + self.b
    
    #4、计算误差
    def calError(self,i):
        Ei = self.distHyperplane(i) - self.Ylabel[i]
        return Ei
    
    #5、SMO算法中第二个参数的修剪
    def cut(self,alpha2New,L,H):
        if alpha2New < L:
            alpha2New = L
        elif alpha2New > H:
            alpha2New = H
        else:
            pass
        return alpha2New
    
    #6、SMO算法中第二个参数上下界的选择
    def LandH(self,y1,y2,alpha1Old,alpha2Old):
        if y1 != y2:
            L = max(0, alpha2Old-alpha1Old)
            H = min(self.C, self.C+alpha2Old-alpha1Old)
        else:
            L = max(0, alpha2Old+alpha1Old-self.C)
            H = min(self.C, alpha2Old+alpha1Old)
#        print('L:',L,'H:',H)
        return L, H
    
    #7、初始化参数
    def initParameter(self):
        self.alpha = np.zeros((self.m,1))
        self.EiMark = np.zeros((self.m,1))
        self.K = np.zeros((self.m,self.m))
        for i in range(self.m):
            self.K[i,:] = self.kernel(self.Xdata, self.Xdata[i])
        return
    
    #8-1、寻找第二个参数，寻找Max(E1-E2)，方法1：随机选取第一个E2
    def choosei2(self,i1):
        self.EiMark[i1] = 1
        E1 = self.calError(i1)
        maxDeltaE = 0
        besti2 = 0
        if np.sum(self.EiMark, axis=0) == 1:
            while True:
                besti2 = np.random.randint(0,self.m)
                if besti2 != i1:
                    self.EiMark[besti2] = 1
                    print('随机选取{}'.format(besti2))
                    break
        else:
            for i in range(self.m):
                if i == i1:
                    continue
                E2 = self.calError(i)
                DeltaE = abs(E1-E2)
                if DeltaE > maxDeltaE:
                    maxDeltaE = DeltaE
                    besti2 = i
        print('最大差法{}'.format(besti2))
        return besti2
        
    
    #8-2、寻找第二个参数，寻找MAX(E1-E2)，方法2：基本固定下来
#    def choosei2(self,i1):
#        E1 = self.calError(i1)
#        maxDeltaE = 0
#        besti2 = 0
#        for i in range(self.m):
#            if i == i1:
#                continue
#            E2 = self.calError(i)
#            DeltaE = abs(E1-E2)
#            if DeltaE > maxDeltaE:
#                maxDeltaE = DeltaE
#                besti2 = i
#            if i1<5:
#                print('E1:',E1,'E2:',E2,'besti2:',besti2)
#        return besti2
    
    #9、内循环：更新迭代alpha1和alpha2参数，和截距b
    def alphaUpdate(self,i1, i2):
        #9-1 两个样本的标签
        y1 = self.Ylabel[i1]
        y2 = self.Ylabel[i2]
        #9-2 参数
        alpha1Old = self.alpha[i1,0].copy()
        alpha2Old = self.alpha[i2,0].copy()
        #9-3 误差
        E1 = self.calError(i1)
        E2 = self.calError(i2)
        #9-4 核转换后的结果
        K11 = self.K[i1,i1]
        K22 = self.K[i2,i2]
        K12 = self.K[i1,i2]
        #结束条件1：
        
        if (K11+K22-2*K12)<=0: return 0
        #9-5 第二个参数的上下界
        L, H = self.LandH(y1,y2,alpha1Old,alpha2Old)
        #结束条件2：
        
        if L == H: return 0
        #9-6 更新第二个参数
        alpha2New = alpha2Old + y2*(E1-E2)/(K11+K22-2*K12)
        #9-7 剪切第二个参数
        alpha2New = self.cut(alpha2New,L,H)
        #结束条件3：
#        print(3)
        if abs(alpha2New - alpha2Old) < 1.0e-5: return 0
        #9-8 更新第一个参数
        alpha1New = alpha1Old + (alpha2Old-alpha2New)*y1*y2
        self.alpha[i1] = alpha1New; self.alpha[i2] = alpha2New
        #9-9 更新截距b
        b1 = self.b - E1 - (alpha1New-alpha1Old)*y1*K11 - (alpha2New-alpha2Old)*y2*K12
        b2 = self.b - E2 - (alpha2New-alpha2Old)*y1*K22 - (alpha1New-alpha1Old)*y2*K12
        if (0<alpha1New and alpha1New<self.C):
            self.b = b1
        elif (0<alpha2New and alpha2New<self.C):
            self.b = b2
        else:
            self.b=(b1+b2)/2
        #结束条件4：
#        print('旧alpha1',alpha1Old,'旧alpha2',alpha2Old,'新alpha1',alpha1New,'新alpha2',alpha2New,'b',self.b)
        return 1
    
    #10、主函数：选择第一个变量进行循环
    def train(self, X, Y, C, kernel, kernalParameter, maxIters=10, tol=0.001):
        #10-1对属性进行赋值
        self.Kvalue[kernel] = kernalParameter
        self.maxiters = maxIters
        self.tol = tol
        self.C = C
        self.Xdata = X
        self.Ylabel = Y
        m, n =np.shape(X)
        self.m = m
        #10-2初始化参数，拉格朗日乘子、核技巧结果等
        self.initParameter()     
        #10-3选择第一个参数
        flag = True             #第一个参数是否遍历全部数据集
        mark = 0                #拉格朗日乘子修改次数
        step = 0                #迭代次数
        while (step<self.maxiters) and (mark>0 or flag):
            mark = 0
            if flag:
                SVMidx = list(range(m))                                         #2、若(0,C)之间的点都符合KKT条件，则遍历全部
                for idx in SVMidx:
                    alpha1 = self.alpha[idx,0]
                    Gi = self.distHyperplane(idx)
                    yi = self.Ylabel[idx,0]
                    if (alpha1==0 and (yi*Gi+self.tol)<1) or (alpha1==self.C and (yi*Gi+self.tol)>1) or (alpha1>0 and alpha1<self.C and (yi*Gi)!=1):
                        i1 = np.copy(idx)                       #找到第1个参数
                        i2 = self.choosei2(i1)                       #找到第2个参数                  
#                        print('i1:',i1,'i2:',i2)
#                        print(">>")
                        mark += self.alphaUpdate(i1, i2)        #进行内循环
                step += 1
                print('全量数据集，第{}轮：，内循环{}次'.format(step,mark))
                print('==================================================')
            else:
                SVMidx = np.nonzero((self.alpha>0)*(self.alpha<self.C))[0]      #1、优先寻找(0,C)之间的点，看是否满足KKT条件
                for idx in SVMidx:
                    alpha1 = self.alpha[idx,0]
                    Gi = self.distHyperplane(idx)
                    yi = self.Ylabel[idx,0]
                    if (alpha1==0 and (yi*Gi+self.tol)<1) or (alpha1==self.C and (yi*Gi+self.tol)>1) or (alpha1>0 and alpha1<self.C and (yi*Gi)!=1):
                        i1 = np.copy(idx)                       #找到第1个参数
                        i2 = self.choosei2(i1)                       #找到第2个参数
#                        print('i1:',i1,'i2:',i2)
#                        print(">>")
                        mark += self.alphaUpdate(i1, i2)        #进行内循环
                step += 1
                print('(0,C)数据集，第{}轮：，内循环{}次'.format(step,mark))
                print('==================================================')
            if flag: flag=False                             #切换到(0,C)之间的点
            elif (mark == 0): flag = True                   #切换到全部数据集
        self.svindex = np.nonzero(self.alpha>0)[0]
        self.SVMvects = self.Xdata[self.svindex]
        self.SVMlabels = self.Ylabel[self.svindex]
        
    #11、预测
    def predict(self,testSet):
        return np.sign(self.distHyperplane(testSet))
        
        
        
        
#读取数据集，并画图
with open(r'D:\mywork\test\ML\郑捷《机器学习算法原理与编程实践》第2-9章节的源代码及数据集\支持向量机\svm.txt', 'r') as f:
    d = f.readlines()
data = np.array([[float(comment) for comment in row.split()] for row in d])
X = data[:,:2]
Y = data[:,2].reshape((len(data),1))        
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='D')      
plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r')        
plt.show()

#训练
smo = SMO()        
smo.train(X, Y, 100, 'Gaussian', 3, maxIters=20)
SVMidx = smo.svindex        
SVMvects = smo.SVMvects
SVMlabels = smo.SVMlabels
print(len(SVMidx))


plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b')      
plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r')
plt.scatter(SVMvects[SVMlabels[:,0]==1,0],SVMvects[SVMlabels[:,0]==1,1],c='g',marker='D',linewidths=5)
plt.scatter(SVMvects[SVMlabels[:,0]==-1,0],SVMvects[SVMlabels[:,0]==-1,1],c='y',marker='D',linewidths=5)        
plt.show()
        
n=0
for i in range(200):
    pre = np.sign(smo.distHyperplane(i))
    if pre!=smo.Ylabel[i]:
        n += 1
        print("第{}个，应该是{}".format(i,smo.Ylabel[i]))        
    



