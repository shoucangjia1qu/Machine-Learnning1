# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:46:07 2020

@author: ecupl
"""

import numpy as np
import pandas as pd

#一、过滤式选择
##Relief(Relevant Features)方法
class Relevant_feature(object):
    #0、属性
    def __init__(self):
        self.nearList = []          #每次抽取样本对应的不同类别的K个最近邻样本下标
        self.sampleList = []        #每次抽取样本的下标
        self.W = 0                  #统计量
        self.trainSet = 0           #原始数据集
        self.normSet = 0            #[0,1]规范化后的数据集
        self.yLabel = 0             #数据标签
        self.yPercentage = {}       #每个Y类别对应的概率
        self.yhitcorrs = {}         #每个猜对近邻对应的概率系数
        self.nsamples = 0           #样本数量
        self.nfeatures = 0          #特征数量
        self.K = 0                  #近邻数量
        self.columns = []           #列名
        self.cateMarks = []         #列是否分类变量

    #1、初始化参数
    def initParas(self, X, Y, cateMarks):
        """
        input：数据集、分类标签
        action：规范化数据、保存分类标签集合、求各类数据占比
        """
        self.nsamples, self.nfeatures = X.shape
        self.trainSet = X
        self.yLabel = Y
        self.normSet = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
        self.yPercentage, self.yhitcorrs = self.calyPercentage(Y)
        self.columns = list(X.columns)
        self.cateMarks = cateMarks
        self.W = np.zeros(self.nfeatures)
        return
    
    #2、计算Y值每类的概率
    def calyPercentage(self, Y):
        """
        输入：分类标签
        输出：分类标签的值集合、除去自身类，其他类的占比
        """
        yPercentage = {}
        yhitcorrs = {}
        m = Y.size
        yset = np.unique(Y)
        for i in yset:
            yPercentage[i] = sum(Y==i)/m
            yhitcorrs[i] = []
            for j in yset:
                if i == j:
                    yhitcorrs[i].append(-1)
                else:
                    yhitcorrs[i].append(sum(Y==j)/(m-sum(Y==i)))
        return yPercentage, yhitcorrs

    #3、计算每个Xi样本在每个Yi类别里的K个最近邻样本
    def calNearArray(self, sampleIdx, sampleY, K):
        """
        sampleIdx: 需要求最近邻的样本下标
        sampleY: 样本的类别
        K: 需要求K个最近邻
        
        """
        self.K = K
        self.sampleList.append(sampleIdx)
        KNDict = {}
        distArr = np.linalg.norm((self.normSet-self.normSet.iloc[sampleIdx,:]), axis=1)
        #print('X下标是：',sampleIdx)
        #print('Y值是：',sampleY)
        #print('距离是：',distArr)
        #print('=============================')
        for yi in self.yPercentage.keys():
            yiIdx = np.nonzero(self.yLabel == yi)[0]
            yidistArr = distArr[yiIdx]
            yisortIdx = list(yiIdx[yidistArr.argsort()])
            if yi == sampleY:
                yisortIdx.remove(sampleIdx)
            KNDict[yi] = yisortIdx[:K]
            #print("yi是：", yi)
            #print("距离下标排序：", yisortIdx)
            #print("最近邻的字典：", KNDict)
            #print("----------------------")
        self.nearList.append(KNDict)
        return KNDict

    #4、求单个抽样样本对应的各个特征的统计量
    def calsampleW(self, KNDict, sampleIdx, sampleY):
        wi = np.zeros(self.nfeatures)
        for colIdx, colName in enumerate(self.columns):
            sigmaKN = []
            if self.cateMarks[colIdx] == 0:
                for KNIdx in KNDict.values():
                    deltaX = (self.normSet.iloc[sampleIdx,colIdx]-self.normSet.iloc[KNIdx,colIdx]).values
                    sigmaKN.append(sum(np.power(deltaX,2)))
            elif self.cateMarks[colIdx] == 1:
                for KNIdx in KNDict.values():
                    deltaX = (self.normSet.iloc[sampleIdx,colIdx] - self.normSet.iloc[KNIdx,colIdx]).values
                    deltaX[np.nonzero(deltaX!=0)[0]] = 1
                    sigmaKN.append(sum(np.power(deltaX,2)))
            else:
                raise ValueError("cateMarks must be 0 or 1 !")
            #print("【%s】"%colName)
            #print("汇总值是：", sigmaKN)
            #print("系数是：", self.yhitcorrs.get(sampleY))
            wi[colIdx] = np.dot(self.yhitcorrs.get(sampleY), sigmaKN)
            #print("wi是", wi)
        return wi

    #5、求最终变量的相关量
    def train(self, X, Y, cateMarks):
        self.initParas(X, Y, cateMarks)
        W = np.zeros(self.nfeatures)
        #判断近邻个数
        if np.unique(Y).__len__() == 2:
            K = 1
        else:
            K = 3
        #判断抽样比例，不超过10000次抽样
        if self.nsamples <= 10000:
            samplingNum = self.nsamples
            for sampleIdx in range(samplingNum):
                sampleY = Y.values[sampleIdx]
                KNDict = self.calNearArray(sampleIdx, sampleY, K)
                wi = self.calsampleW(KNDict, sampleIdx, sampleY)
                W += wi
        else:
            samplingNum = 10000
            for i in range(samplingNum):
                sampleIdx = np.random.randint(0, self.nsamples)
                sampleY = Y.values[sampleIdx]
                KNDict = self.calNearArray(sampleIdx, sampleY, K)
                wi = self.calsampleW(KNDict, sampleIdx, sampleY)
                W += wi        
        
        self.W = W/(samplingNum*K)
        return
    
#%%            
###训练测试
if __name__ == "__main__":
    from sklearn import datasets
    dt = datasets.load_iris()
    trainSet = dt.data
    X = pd.DataFrame(trainSet)
    Label = dt.target
    Y = pd.Series(Label)
    relief = Relevant_feature()
    relief.train(X, Y, [0,0,0,0])
    W = relief.W                        #各属性的统计量
    normX = relief.normSet              #[0,1]规范化后的数据集
    sampleList = relief.sampleList      #样本下标
    nearList = relief.nearList          #各个样本各类别中的近邻样本下标
    Ycorrs = relief.yhitcorrs

    
    
    
    
#鸢尾花数据测试情况
'''    
X下标是： 142
Y值是： 2
距离是： [1.02173072 0.98647803 1.02631692 1.00644513 1.04164371 0.99250147
 1.01681011 1.00423856 1.02625188 1.01246461 1.02956287 1.00790695
 1.02399379 1.10062443 1.10999203 1.12491756 1.03334119 0.99329854
 0.99366693 1.03180877 0.96500762 0.98709248 1.111926   0.87901928
 0.97884339 0.9584645  0.93647612 1.00644513 1.00392287 0.99519519
 0.97907342 0.92754888 1.14142996 1.13691836 0.98199832 1.01561066
 1.01716026 1.07652348 1.04087415 0.99845932 1.00983246 1.00868567
 1.05413319 0.89766752 0.96574915 0.96462829 1.04953541 1.02446464
 1.03292989 1.00344239 0.4500145  0.33060579 0.38738721 0.36329375
 0.2729549  0.27449876 0.3193924  0.55841445 0.35497747 0.33548171
 0.59043988 0.25969898 0.47103537 0.24877308 0.37036126 0.38439169
 0.23839265 0.41152446 0.306379   0.40312454 0.22021798 0.3254853
 0.23486473 0.31360226 0.34001139 0.3499813  0.3533905  0.29215872
 0.21943222 0.46548341 0.42688385 0.46850838 0.35557962 0.13678969
 0.25708036 0.33782054 0.35021698 0.35163094 0.3315702  0.33339018
 0.32836933 0.27047008 0.34866271 0.55755017 0.29808656 0.35317874
 0.30575354 0.31650536 0.53155614 0.30616187 0.40934008 0.
 0.41395193 0.18749006 0.28833151 0.58069445 0.29450084 0.47292609
 0.29198796 0.61894018 0.28800613 0.17007902 0.32299585 0.09868857
 0.21245915 0.31639758 0.24447113 0.7601223  0.63335612 0.27304624
 0.41819446 0.08779372 0.59629162 0.14891383 0.37720856 0.46866047
 0.135659   0.15954514 0.20890232 0.44335199 0.47748754 0.7750049
 0.22873726 0.2209163  0.2434444  0.59218201 0.39362951 0.24877308
 0.15176625 0.36148566 0.3753164  0.38590122 0.         0.40832165
 0.44479417 0.32586815 0.16285528 0.23549316 0.357461   0.13465777]
=============================
yi是： 0
距离下标排序： [23, 43, 31, 26, 25, 45, 20, 44, 24, 30, 34, 1, 21, 5, 17, 18, 29, 39, 49, 28, 7, 3, 27, 11, 41, 40, 9, 35, 6, 36, 0, 12, 47, 8, 2, 10, 19, 48, 16, 38, 4, 46, 42, 37, 13, 14, 22, 15, 33, 32]
最近邻的字典： {0: [23, 43, 31]}
----------------------
yi是： 1
距离下标排序： [83, 78, 70, 72, 66, 63, 84, 61, 91, 54, 55, 77, 94, 96, 99, 68, 73, 97, 56, 71, 90, 51, 88, 89, 59, 85, 74, 92, 75, 86, 87, 95, 76, 58, 82, 53, 64, 65, 52, 69, 67, 80, 50, 79, 81, 62, 98, 93, 57, 60]
最近邻的字典： {0: [23, 43, 31], 1: [83, 78, 70]}
----------------------
yi是： 2
距离下标排序： [101, 121, 113, 149, 126, 123, 138, 127, 146, 111, 103, 128, 114, 133, 132, 147, 134, 116, 137, 119, 110, 104, 108, 106, 115, 112, 145, 148, 139, 140, 124, 141, 136, 143, 100, 102, 120, 129, 144, 125, 107, 130, 105, 135, 122, 109, 118, 117, 131]
最近邻的字典： {0: [23, 43, 31], 1: [83, 78, 70], 2: [101, 121, 113]}
----------------------
【0】
汇总值是： [0.09953703703703701, 0.006944444444444457, 0.003858024691358028]
系数是： [0.5, 0.5, -1]
wi是 [0.04938272 0.         0.         0.        ]
【1】
汇总值是： [0.25868055555555536, 0.050347222222222196, 0.008680555555555549]
系数是： [0.5, 0.5, -1]
wi是 [0.04938272 0.14583333 0.         0.        ]
【2】
汇总值是： [1.0563056592933062, 0.012927319735708117, 0.0014363688595231178]
系数是： [0.5, 0.5, -1]
wi是 [0.04938272 0.14583333 0.53318012 0.        ]
【3】
汇总值是： [1.0243055555555554, 0.04513888888888887, 0.003472222222222216]
系数是： [0.5, 0.5, -1]
wi是 [0.04938272 0.14583333 0.53318012 0.53125   ]
'''





