# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:04:34 2018

@author: ecupl
"""

import numpy as np
import pandas as pd
import os,copy
import matplotlib.pyplot as plt

#######################
#                     #
#        SVM          #
#                     #
#######################
os.chdir(r"D:\mywork\test\ML")

import copy
class PlattSVM(object):
    def __init__(self):
        self.trainSet = 0       #数据集
        self.Labels = 0         #标签
        self.K = 0              #经核函数转变后的点积
        self.kValue = dict()    #核函数的参数
        self.C = 0              #惩罚因子C
        self.alpha = 0          #拉格朗日乘子
        self.tol = 0            #容错率
        self.maxiters = 100     #最大循环次数
        self.b = 0              #截距初始值
        
        
    
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
        self.eCache = np.zeros((m,2))          #一列为标注误差是否更新，一列是记载的误差
        self.K = np.zeros((m,m))
        for k in range(m):
            self.K[k,:] = self.kernels(self.trainSet,self.trainSet[k,:])
    
    '''构造核函数'''
    def kernels(self,data,A):
        if list(self.kValue.keys())[0] == "linear":
            Ki = np.dot(data,A.T)
        elif list(self.kValue.keys())[0] == "Gaussian":
            x = np.power((data - A),2)
            Mo = np.sum(x,axis=1)
            Ki = np.exp(Mo/(-1*self.kValue['Gaussian']**2))
        else:
            raise NameError('无法识别的核函数')
        
        return Ki

    '''计算误差函数'''
    def calEk(self,i):
        Ek = float(np.dot(np.multiply(self.alpha,self.Labels).T,self.K[:,i])) + self.b - self.Labels[i,0]      # Yp=W*X+b;E=Yp-Y
        return Ek
    
    '''选择子循环的变量'''
    def chooseJ(self,i,Ei):
        maxj = -1
        maxEj = 0
        maxDeltaE = 0
        self.eCache[i] = (1,Ei)
        SvmError = np.nonzero(self.eCache[:,0])[0]         #判断当前误差的个数，如果只有一个说明是第一次，就可以随机选取J
        if len(SvmError)>1:
            '''目前不止一个误差'''
            for j in SvmError:
                if j==i:
                    continue
                Ej = self.calEk(j)                   #重新遍历第二个变量的误差
                deltaE = abs(Ei-Ej)                  #第一个和第二个变量的误差差
                if deltaE>maxDeltaE:
                    maxj = j                        #最终选取的第二个变量
                    maxEj = Ej                      #最终选取的第二个变量的误差
                    maxDeltaE = deltaE
            return maxj,maxEj
        else:
            '''目前只有一个误差'''
            while True:
                j = np.random.randint(0,np.shape(self.trainSet)[0])
                if j != i :
                    break
            Ej = self.calEk(j)
            return j,Ej
    
    def cutalpha(self,alpha,L,H):
        if alpha>H:
            alpha = H
        if alpha<L:
            alpha = L
        return alpha
    
    '''主函数：主循环'''
    def train(self):
        self.initparam()        #初始化
        m,n = np.shape(self.trainSet)
        step = 0                #循环次数
        flag = True             #主循环标识
        AlphaChange = 0         #内循环标识
        while step<self.maxiters and (flag or AlphaChange>0):
            AlphaChange = 0         #内循环标识
            if flag:
                for i in range(m):                          #2、其次遍历全量数据集
                    Ei = self.calEk(i)
                    alpha1 = self.alpha[i,0]
                    y1 = self.Labels[i,0]
                    if (alpha1==0 and y1*Ei<-self.tol) or (alpha1==self.C and y1*Ei>self.tol) or (alpha1>0 and alpha1<self.C and y1*Ei!=0):
                        AlphaChange += self.inner(i,Ei,y1)        #内循环返还标识
                print("全量",step,AlphaChange)
                step+=1
            else:
                SvmAlpha = np.nonzero((self.alpha>0)*(self.alpha<self.C))[0]        #1、优先查找（0，C）之间的拉格朗日乘子
                for i in SvmAlpha:
                    Ei = self.calEk(i)
                    alpha1 = self.alpha[i,0]
                    y1 = self.Labels[i,0]
                    if (alpha1==0 and y1*Ei<-self.tol) or (alpha1==self.C and y1*Ei>self.tol) or (alpha1>0 and alpha1<self.C and y1*Ei!=0):
                        '''(alpha1<self.C and y1*Ei<-self.tol) or (alpha1>0 and y1*Ei>self.tol)'''
                        '''判定是否符合KKT条件，不符合的就进行内循环'''
                        AlphaChange += self.inner(i,Ei,y1)        #内循环返还标识
                print("KKT",step,AlphaChange)
                step+=1
            if flag : flag = False                                                                                          #转换标志位：切换到另一种
            elif (AlphaChange == 0) :flag = True                            #改变主循环数据集为（0，C）
        self.svIdx = np.nonzero(self.alpha>0)[0]            #支持向量的下标
        self.sptVects = self.trainSet[self.svIdx]           #支持向量
        self.SVlabels = self.Labels[self.svIdx]             #支持向量的标签
        print(step)

    '''主函数：内循环'''
    def inner(self,i,Ei,y1):
        j,Ej = self.chooseJ(i,Ei)           #生成第二个变量
        y2 = self.Labels[j,0]
        oldAlpha1 = self.alpha[i,0].copy()     #生成旧的第一个alpha
        oldAlpha2 = self.alpha[j,0].copy()     #生成旧的第二个alpha
        '''进行剪枝条件的判定'''
        if y1!=y2:
            L=max(0,oldAlpha2-oldAlpha1)
            H=min(self.C,self.C+oldAlpha2-oldAlpha1)
        else:
            L=max(0,oldAlpha2+oldAlpha1-self.C)
            H=min(self.C,oldAlpha2+oldAlpha1)
        if L==H:
            return 0
        '''求解新的alpha变量'''
        eta = self.K[i,i] + self.K[j,j] - 2*self.K[i,j]
        if eta<=0:
            return 0
        '''未剪枝的newAlpha2'''
        self.alpha[j,0] = oldAlpha2 +y2*(Ei-Ej)/eta
        '''选定最终的Alpha2'''
        self.alpha[j,0] = self.cutalpha(self.alpha[j,0],L,H)
        '''计算最终的Alpha1'''
        if abs(oldAlpha2-self.alpha[j,0])<0.00001:
            return 0
        self.alpha[i,0] = oldAlpha1 + y1*y2*(oldAlpha2-self.alpha[j,0])
        '''计算最终的误差Ei和Ej'''
        self.eCache[j] = (1,self.calEk(j))
        self.eCache[i] = (1,self.calEk(i))
        '''计算最终的b'''
        b1 = self.b-Ei-y1*self.K[i,i]*(self.alpha[i,0]-oldAlpha1)-y2*self.K[j,i]*(self.alpha[j,0]-oldAlpha2)
        b2 = self.b-Ej-y1*self.K[i,j]*(self.alpha[i,0]-oldAlpha1)-y2*self.K[j,j]*(self.alpha[j,0]-oldAlpha2)
        if (0<self.alpha[i,0] and self.alpha[i,0]<self.C):
            self.b = b1
        elif (0<self.alpha[j,0] and self.alpha[j,0]<self.C):
            self.b = b2
        else:
            self.b=(b1+b2)/2
        return 1

    def scatterplot(self, plt):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(shape(self.trainSet)[0]) :
            if self.alpha[i] != 0 :
                ax.scatter(self.trainSet[i, 0], self.trainSet[i, 1], c = 'green', marker = 's')
            elif self.Labels[i] == 1 :
                ax.scatter(self.trainSet[i, 0], self.trainSet[i, 1], c = 'blue', marker = 'o')
            elif self.Labels[i] == -1 :
                ax.scatter(self.trainSet[i, 0], self.trainSet[i, 1], c = 'red', marker = 'o')

    
    '''预测'''
    def predict(self,testSet):
        m,n = np.shape(testSet)
        preLabels = np.zeros([m,1])
        for i in range(m):
            sigmaK = self.kernels(self.sptVects,testSet[i,:])
            preY = np.multiply(self.alpha[self.svIdx],self.SVlabels).T*sigmaK + self.b
            preLabels[i,0] = np.sign(preY)
        return preLabels

    def classify(self, testSet, testLabel) :
        errorCount = 0
        testMat = mat(testSet)
        m, n = shape(testMat)
        for i in range(m) :
            kernelEval = self.kernels(self.sptVects, testMat[i, :])
            predict = kernelEval.T * multiply(self.SVlabels, self.alpha[self.svIdx]) + self.b
            if sign(predict) != sign(testLabel[i]) : errorCount += 1
        return float(errorCount) / float(m)



svm = PlattSVM()
svm.C = 100                                                                                   #惩罚因子
svm.tol = 0.001                                                                            #容错律
svm.maxiters = 10
svm.kValue['Gaussian'] = 3.0                                                  #核函数
svm.loadData('svm.txt')
svm.train()
svm.scatterplot(plt)
plt.show()
print(svm.classify(svm.trainSet, svm.Labels))
print(svm.svIdx)
print(shape(svm.sptVects)[0])
print("b: ", svm.b)

'''[  3   9  26  27  33  40  49  52  53  59  79  81 100 101 102 103 104 106
 107 109 130 133 148 198 199]
25
b:  [[ 8.54969697]]'''











