# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:39:48 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy

os.chdir(r'D:\\mywork\\test')

#数据集准备
from sklearn.preprocessing import OrdinalEncoder
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
X=np.array(dataSet)[:,6:8].astype(float)
Y = np.array(dataSet)[:,8]
Y[Y=="好瓜"]=1
Y[Y=="坏瓜"]=-1
Y=Y.astype(float)
Y = Y.reshape(-1,1)


#########Adaboost集成算法#########
#######写个决策树，单层决策树#######
class dtree(object):
    #属性
    def __init__(self):
        self.treeDict = {}              #树结构
        self.treeDepth = 0              #树深
        self.Gain_Ratio = {}            #信息增益率的特征和阀值
        self.LabelName = []             #特征名称
    
    #计算信息熵
    def __entropy(self, y, w):
        ent = 0
        for value in np.unique(y):
            xIdx = np.nonzero(y==value)[0]
            p = np.sum(w[xIdx])
            ent+=-(p)*np.log2(p)
        return ent
    
    #计算信息增益率
    def __getGainRatio(self, x, y, columnIndex, w):
        Gain = 0
        Iv = 0                          #属性固有值
        Entropy0 = self.__entropy(y, w) #计算总信息熵
        bestThreshold = None            #最优划分点
        bestGainRatio = 0               #最大信息增益率
        m, n = np.shape(x)
        if columnIndex > (n-1):
            raise ValueError('Max columnIndex is %d !'%(n-1))
        else:
            x = x[:,columnIndex]
        #判断是分类还是连续变量
        xValue = np.unique(x)
        if len(xValue) <= 3:
            ##计算各属性的样本占比
            Di = [np.sum(w[np.nonzero(x==value)[0]]) for value in xValue]
            ##计算各属性对应样本的信息熵
            Enti = [self.__entropy(y[np.nonzero(x==value)[0]], w[np.nonzero(x==value)[0]]) for value in xValue]
            ##计算信息增益
            Gain = Entropy0 - np.dot(Di, Enti)
            ##计算Iv固有值
            Iv = self.__entropy(x, w)
            ##当Iv=0时，属性值一样信息熵为0，不为0再计算增益率
            bestGainRatio = 0 if Iv==0 else Gain/Iv
        else:
            thresholds = [(xValue[i]+xValue[i+1])/2 for i in range(len(xValue)-1)]
            for thres in thresholds:
                Di = [np.sum(w[np.nonzero(x<=thres)[0]]), np.sum(w[np.nonzero(x>thres)[0]])]
                Enti = [self.__entropy(y[np.nonzero(x<=thres)[0]], w[np.nonzero(x<=thres)[0]]), self.__entropy(y[np.nonzero(x>thres)[0]], w[np.nonzero(x>thres)[0]])]
                Gain = Entropy0 - np.dot(Di, Enti)
                Iv = -np.dot(Di, np.log2(Di))
                Gain_Ratio = Gain/Iv
                if Gain_Ratio > bestGainRatio:
                    bestGainRatio = Gain_Ratio
                    bestThreshold = thres
        return bestGainRatio, bestThreshold
    
    #判断数据集对应的类
    def __chooseLabel(self, y, w):
        ylabel = None
        ycount = 0
        for value in np.unique(y):
            count = np.sum(w[np.nonzero(y==value)[0]])
            #若正反样本数量相同，以Yes为基准
            if count >= ycount:
                ycount = count
                ylabel = value
        return ylabel
    
    #递归函数生成树结构
    def buildTree(self, x, y, treeDepth, w):
        self.initparas()
        propertySet = list(range(len(self.LabelName)))
        self.treeDepth = treeDepth
        maxGainRatio = 0
        threshold = None
        bestProperty = None
#        leafDict = {}
#        treeDict = {}
        #结束条件1：样本标签相同，返回当前数据的分类
        if len(np.unique(y)) == 1:
            return self.__chooseLabel(y, w)
        #结束条件2：无属性值可分，或者所有样本属性值相同，返回当前数据的分类
        if (propertySet is None) or len(np.unique(x[:,propertySet], axis=0))==1:
            return self.__chooseLabel(y, w)
        #选取特征集合中最优的特征
        for i in propertySet:
            GainRatio, thres = self.__getGainRatio(x, y, i, w)
            if GainRatio > maxGainRatio:
                maxGainRatio = GainRatio
                threshold = thres
                bestProperty = i
        #用字典记录每一次的分类特征与信息增益率、阈值
        self.Gain_Ratio[self.LabelName[bestProperty]] = threshold
        #复制子特征集合
#        subpropertySet = np.copy(propertySet).tolist()
        #如果选择预剪枝，则触发结束条件3：判断是否剪枝
        #if self.mode == 'pre':
            ##（1）深度复制树结构
        #    afterTree = copy.deepcopy(self.treeDict)
            ##（2）判断是分类还是连续变量，分别追加叶节点
        #    leafDict[self.LabelName[bestProperty]] = dict()
        #    if threshold is None:
        #        for xvalue in np.unique(x[:,bestProperty]):
        #            subx, suby = self.__splitDataset(x, y, bestProperty, xvalue)
        #            leafDict[self.LabelName[bestProperty]][xvalue] = self.__chooseLabel(y) if len(suby)==0 else self.__chooseLabel(suby)
        #    else:
        #        for marker in ['less', 'more']:
        #            subx, suby = self.__splitDataset(x, y, bestProperty, threshold, marker)
        #            leafDict[self.LabelName[bestProperty]][marker+'/'+str(threshold)] = self.__chooseLabel(y) if len(suby) == 0 else self.__chooseLabel(suby)
            ##(3)将叶节点加到当前树中
        #    afterTree = leafDict if afterTree == dict() else self.__api(afterTree,treeIndex,leafDict)
            ##(4)判断是否剪枝，剪枝的话返回上一节点的类别，不剪枝的话替代当前树结构。
        #    pruningSign = self.__pruning(self.testX, self.testY, self.treeDict, afterTree)
        #    if pruningSign == 1:            
        #        return self.__chooseLabel(y)
        #    else:                           
        #        self.treeDict = afterTree
        #不符合结束条件和剪枝，可以继续递归
#        treeIndex.append(self.LabelName[bestProperty])              #完善树的节点
        if threshold is None:
            for xvalue in np.unique(x[:,bestProperty]):
                xIdx = np.nonzero(x[:,bestProperty]==xvalue)[0]
                suby = y[xIdx]
                subw = w[xIdx]
                subTree = self.__chooseLabel(suby, subw)
                if self.LabelName[bestProperty] not in self.treeDict.keys():
                    self.treeDict[self.LabelName[bestProperty]] = dict()
                self.treeDict[self.LabelName[bestProperty]][xvalue] = subTree
        else:
            for marker in ['less', 'more']:
                if marker == 'less':
                    xIdx = np.nonzero(x[:,bestProperty]<=threshold)[0]
                else:
                    xIdx = np.nonzero(x[:,bestProperty]>threshold)[0]
                suby = y[xIdx]
                subw = w[xIdx]
                subTree = self.__chooseLabel(suby, subw)
                if self.LabelName[bestProperty] not in self.treeDict.keys():
                    self.treeDict[self.LabelName[bestProperty]] = dict()
                self.treeDict[self.LabelName[bestProperty]][marker+'/'+str(threshold)] = subTree
        return self.treeDict, self.Gain_Ratio

    def initparas(self):
        self.treeDict = {}
        self.Gain_Ratio = {}
        self.LabelName = ['密度', '含糖度']
        return

    def predict(self, x):
        m,n = np.shape(x)
        preY = np.zeros((m,1))
        for i in range(m):
            for node in self.treeDict.keys():
                xIdx = self.LabelName.index(node)
                value = x[i,xIdx]
                for leaf in self.treeDict[node].keys():
                    if not isinstance(leaf, str):
                        preY[i,0] = self.treeDict[node][value]
                    else:
                        marker, threshold = leaf.split('/')
                        threshold = float(threshold)
                        if (marker=='less') and (value<=threshold):
                            preY[i,0] = self.treeDict[node][leaf]
                        elif (marker=='more') and (value>threshold):
                            preY[i,0] = self.treeDict[node][leaf]
        return preY

######编写AdaBoost算法
class AdaBoost(dtree):
    def __init__(self):
        super().__init__()
        self.Gx = {}                #集合分类器
        self.errList = []           #分类误差率
        self.wList = []             #每个样本的权重列表
        self.alphaList = []         #每个分类器的权重
        self.GainthresList = []     #每个分类器的属性和阀值
        self.X = 0
        self.Y = 0
        self.iters = 11
        
    def calAlpha(self, err):
        return 0.5*np.log((1-err)/err)
    
    def changeW(self, preY, Y, w0, alpha):
        sumValue = np.sum(np.multiply(w0, np.exp(-alpha*np.multiply(preY, Y))))
        w1 = np.multiply(w0, np.exp(-alpha*np.multiply(preY, Y)))/sumValue
        return w1
    
    def calErr(self, preY, Y, w):
        idx =  np.nonzero(preY != Y)[0]
        return np.sum(w[idx])
    
    def train(self, X, Y):
        self.X = X
        self.Y = Y
        m, n =np.shape(X)
        w1 = np.ones((m,1))/m
        self.wList.append(w1)
        for i in range(self.iters):
            treeDict, Gainthres = self.buildTree(self.X, self.Y, 1, w1)
            print(treeDict)
            preY = self.predict(self.X)
            err = self.calErr(preY, self.Y, w1)
            alpha = self.calAlpha(err)
            w0 = copy.deepcopy(w1)
            w1 = self.changeW(preY, self.Y, w0, alpha)
            self.errList.append(err)
            self.wList.append(w1)
            if err<0.5:
                self.Gx[alpha] = treeDict
                self.alphaList.append(alpha)
                self.GainthresList.append(Gainthres)
        return




ab = AdaBoost()
ab.train(X,Y)
Gx = ab.Gx
LabelName = ab.LabelName
#预测
def predict(x, treeDict, LabelName):
    m,n = np.shape(x)
    preY = np.zeros((m,1))
    for i in range(m):
        for node in treeDict.keys():
            xIdx = LabelName.index(node)
            value = x[i,xIdx]
            for leaf in treeDict[node].keys():
                if not isinstance(leaf, str):
                    preY[i,0] = treeDict[node][value]
                else:
                    marker, threshold = leaf.split('/')
                    threshold = float(threshold)
                    if (marker=='less') and (value<=threshold):
                        preY[i,0] = treeDict[node][leaf]
                    elif (marker=='more') and (value>threshold):
                        preY[i,0] = treeDict[node][leaf]
    return preY

preYvalue = 0
for value, treeDict in Gx.items():
    preYvalue += value*predict(X, treeDict, LabelName)
preY = np.sign(preYvalue)




Gain = ab.GainthresList
#画图
plt.scatter(X[(Y==1)[:,0],0], X[(Y==1)[:,0],1], c='r', marker='+')
plt.scatter(X[(Y==-1)[:,0],0], X[(Y==-1)[:,0],1], c='b', marker='D')
for i in Gain:
    for fea, thres in i.items():
        if fea=='含糖度':
            plt.plot(list(np.linspace(0,0.8,10)), [thres]*10, c='g')
        else:
            plt.plot([thres]*10, list(np.linspace(0,0.5,10)), c='g')
plt.show()

















