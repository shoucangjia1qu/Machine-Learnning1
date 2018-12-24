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
        self.trainSet = 0
        self.Labels = 0
        self.K = 0              #经核函数转变后的点积
        self.kValue = dict()
        
    
    '''读取数据集'''
    def loadData(self,filename):
        with open(filename,"r") as f:
            content = f.readlines()
            OriData = np.array([[float(comment) for comment in row.split()] for row in content])
        self.trainSet = OriData[:,:2]
        self.Labels = OriData[:,2]
            
     
        
    
    
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
















'''画图展示数据分布'''
plt.figure()
plt.scatter(trainSet[:100,0],trainSet[:100,1],marker='X',c='r',linewidths=1)
plt.scatter(trainSet[100:,0],trainSet[100:,1],marker='o',c='b',linewidths=2)
plt.show()













