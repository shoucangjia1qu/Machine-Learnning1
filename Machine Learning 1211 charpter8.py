# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:04:34 2018

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#######################
#                     #
#        SVM          #
#                     #
#######################
os.chdir(r"D:\mywork\test\ML")

def PlattSVM(object):
    def __init__():
        self.trainSet = 0       #数据集
        self.Labels = 0         #标签
        self.K = 0              #经核函数转变后的点积
        self.kValue = dict()    #核函数的参数
        self.C = 0              #惩罚因子C
        self.alpha = 0          #拉格朗日乘子
        self.tol = 0            #容错率
        self.maxiters = 100     #最大循环次数
        
        
    
    '''读取数据集'''
    def loadData(self,filename):
        with open(filename,"r") as f:
            content = f.readlines()
            OriData = np.array([[float(comment) for comment in row.split()] for row in content])
        self.trainSet = OriData[:,:2]
        self.Labels = OriData[:,2].reshape((len(OriData),1))
    
    '''初始化'''        
     def initparam(self):
         m,n = np.shape(self.trainSet)
         self.alpha = np.zeros((m,1))
         self.
        
    
    
    '''构造核函数'''
    def kernels(self,data):
        m,n = np.shape(data)
        self.K = np.zeros(m,m)
        for i in range(m):
            A = data[i,:]
            if list(self.kValue.keys())[0] == "linear":
                self.K[i,:] = np.dot(data,A.T)
            elif list(self.kValue.keys())[0] == "Gaussian":
                x = np.power((data - A),2)
                Mo = np.sqrt(np.sum(x,axis=1))
                self.K[i,:] = np.exp(Mo/(-2*self.kValue['Gaussian']**2)
            else:
                raise NameError('无法识别的核函数')
        print("核函数转换完毕")
    
    '''计算误差函数'''
    def calEk(self,i):
        Ek = np.multiply(self.alpha,self,Labels).T*self.K[:,i] + self.b - self.Labels[i,0]      # Yp=W*X+b;E=Yp-Y
        return Ek
    
    
    
    '''主函数：主循环'''
    def train(self):
        m,n = np.shape(self.trainSet)
        step = 0                #循环次数
        flag = True             #主循环标识
        AlphaChange = 0         #内循环标识
        while step<self.maxiters and (flag==True or AlphaChange>0):
            if flag:
                SvmAlpha = np.nonzero((self.alpha>0)*(self.alpha<self.C))[0]        #1、优先查找（0，C）之间的拉格朗日乘子
                for i in SvmAlpha:
                    Ei = self.calEk(i)
                    alpha1 = self.alpha[i,0]
                    y1 = self.Labels[i,0]
                    if (alpha1==0 and y1*Ei<-self.tol) or (alpha1==self.C and y1*Ei>self.tol) or (alpha1>0&alpha1<self.C and y1*Ei!=0):
                        '''(alpha1<self.C and y1*Ei<-self.tol) or (alpha1>0 and y1*Ei>self.tol)'''
                        '''判定是否符合KKT条件，不符合的就进行内循环'''
                        AlphaChange += self.inner(i)        #内循环返还标识
                step+=1
                if AlphaChange==0:
                    flag = False                            #改变主循环数据集为全量
            else:
                for i in range(m):                          #2、其次遍历全量数据集
                    Ei = self.calEk(i)
                    alpha1 = self.alpha[i,0]
                    y1 = self.Labels[i,0]
                    if (alpha1==0 and y1*Ei<-self.tol) or (alpha1==self.C and y1*Ei>self.tol) or (alpha1>0&alpha1<self.C and y1*Ei!=0):
                        AlphaChange += self.inner(i)        #内循环返还标识
                    step+=1
                if AlphaChange>0:
                    flag = True                             #改变主循环数据集为（0，C）

    














'''画图展示数据分布'''
plt.figure()
plt.scatter(trainSet[:100,0],trainSet[:100,1],marker='X',c='r',linewidths=1)
plt.scatter(trainSet[100:,0],trainSet[100:,1],marker='o',c='b',linewidths=2)
plt.show()













