# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:02:13 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import os, copy

os.chdir(r'D:\mywork\test')

#数据准备
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
X = np.array(dataSet)[:,:8]
Y = np.array(dataSet)[:,8]

#对X进行编码
from sklearn.preprocessing import OrdinalEncoder
oriencode = OrdinalEncoder(categories='auto')
oriencode.fit(X[:,:6])
Xdata=oriencode.transform(X[:,:6])           #编码后的数据
print(oriencode.categories_)                       #查看分类标签
Xdata=np.hstack((Xdata,X[:,6:].astype(float)))

#对Y进行编码
from sklearn.preprocessing import LabelEncoder
labelencode = LabelEncoder()
labelencode.fit(Y)
Ylabel=labelencode.transform(Y)       #得到切分后的数据
labelencode.classes_                        #查看分类标签
labelencode.inverse_transform(Ylabel)    #还原编码前数据

class EntTree(object):
    #1、初始设置
    def __init__(self):
        self.treeDict = {}          #生成树的规则
        self.trainSet = 0
        self.label = 0
        self.testX = 0
        self.testY = 0
        self.LabelName = 0          #各变量对应的名称
        self.Gain_Ratio = {}        #记录每次迭代的信息增益率
    #2、实例化的展示   
    def __str__(self):
        return 'the tree is Gain_Raito Tree'
    __repr__ = __str__
    
    #3、计算信息熵
    def __entropy(self, y):
        ent = 0
        for value in set(y):
            ent+=-(sum(y==value)/len(y))*np.log2(sum(y==value)/len(y))
        return ent
    
    #4、计算信息增益率
    def __getGainRatio(self, x, y , columnIndex):
        Gain = 0
        Iv = 0                          #属性固有值
        Entropy0 = self.__entropy(y)    #计算总信息熵
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
            Di = [sum(x==value)/len(x) for value in xValue]
            ##计算各属性对应样本的信息熵
            Enti = [self.__entropy(y[np.nonzero(x==value)[0]]) for value in xValue]
            ##计算信息增益
            Gain = Entropy0 - np.dot(Di, Enti)
            ##计算Iv固有值
            Iv = self.__entropy(x)
            ##当Iv=0时，属性值一样信息熵为0，不为0再计算增益率
            bestGainRatio = 0 if Iv==0 else Gain
        else:
            thresholds = [(xValue[i]+xValue[i+1])/2 for i in range(len(xValue)-1)]
            for thres in thresholds:
                Di = [sum(x<=thres)/len(x), sum(x>thres)/len(x)]
                Enti = [self.__entropy(y[np.nonzero(x<=thres)[0]]), self.__entropy(y[np.nonzero(x>thres)[0]])]
                Gain = Entropy0 - np.dot(Di, Enti)
                Iv = -np.dot(Di, np.log2(Di))
                Gain_Ratio = Gain
                if Gain_Ratio > bestGainRatio:
                    bestGainRatio = Gain_Ratio
                    bestThreshold = thres
        return bestGainRatio, bestThreshold
    
    #5、划分新的数据集和标签
    def __splitDataset(self, x, y, column, threshold, marker=None):
        if marker is None:
            subx = x[np.nonzero(x[:,column]==threshold)[0],:]    #选择当前属性值的样本
            suby = y[np.nonzero(x[:,column]==threshold)[0]]      #选择当前属性值的标签
        elif marker == 'less':
            subx = x[np.nonzero(x[:,column]<=threshold)[0],:]
            suby = y[np.nonzero(x[:,column]<=threshold)[0]]
        elif marker == 'more':
            subx = x[np.nonzero(x[:,column]>threshold)[0],:]
            suby = y[np.nonzero(x[:,column]>threshold)[0]]
        return subx, suby
    
    #6、判断数据集对应的类
    def __chooseLabel(self, y):
        ylabel = None
        ycount = 0
        ylist = y.tolist()
        for value in set(y):
            count = ylist.count(value)
            #若正反样本数量相同，以Yes为基准
            if count >= ycount:
                ycount = count
                ylabel = value
        return ylabel
    
    #7、修改指定树节点的叶子
    def __api(self, tree,Index,value,n=0):
        if n<len(Index)-1:
            newtree = tree[Index[n]]
            n += 1
            self.__api(newtree,Index,value,n)
        else:
            tree[Index[n]] = value
        return tree

    #8、根据树结构对样本进行预测
    def __predictTree(self, x, tree):
        for node in tree.keys():
            column = self.LabelName.index(node)
            value = x[column]
            for leaf in tree[node].keys():
                #print('预测Y:',tree[node][leaf])
                if isinstance(leaf, int) or isinstance(leaf, float):
                    if (value==leaf) and (not isinstance(tree[node][value], dict)):
                        return tree[node][value]
                    elif (value==leaf) and (isinstance(tree[node][value], dict)):
                        return self.__predictTree(x, tree[node][leaf])
                elif isinstance(leaf, str):
                    marker, threshold = leaf.split('/')
                    threshold = float(threshold)
                    if marker=='less':
                        if (value<=threshold) and (not isinstance(tree[node][leaf], dict)):
                            return tree[node][leaf]
                        elif (value<=threshold) and (isinstance(tree[node][leaf], dict)):
                            return self.__predictTree(x, tree[node][leaf])
                    elif marker=='more':
                        if (value>threshold) and (not isinstance(tree[node][leaf], dict)):
                            return tree[node][leaf]
                        elif (value>threshold) and (isinstance(tree[node][leaf], dict)):
                            return self.__predictTree(x, tree[node][leaf])

    #9、判断是否需要剪枝
    def __pruning(self, testX, testY, preTree, afterTree):
        m,n = np.shape(testX)
        #计算剪枝的准确率
        acc0 = 0
        #初始树，即不划分
        if preTree == dict():
            result = self.__chooseLabel(self.label)
            acc0 = sum(testY == result)/len(testY)
        else:
            for i in range(m):
                preY = self.__predictTree(testX[i,:], preTree)
                if preY==testY[i]:
                    acc0+=1
            acc0/=len(testY)
        print('剪枝的准确率：',acc0)
        #计算不剪枝的准确率
        acc1 = 0
        for i in range(m):
            preY = self.__predictTree(testX[i,:], afterTree)
            if preY==testY[i]:
                acc1+=1
        acc1/=len(testY)
        print('不剪枝的准确率：',acc1)
        if acc1>acc0:
            return 0    
        else:
            return 1

    #10、递归函数生成树结构
    def buildTree(self, x, y, propertySet, treeIndex=[]):
        maxGainRatio = 0
        threshold = None
        bestProperty = None
        leafDict = {}
        treeDict = {}
        #结束条件1：样本标签相同，返回当前数据的分类
        if len(np.unique(y)) == 1:
            return self.__chooseLabel(y)
        #结束条件2：无属性值可分，或者所有样本属性值相同，返回当前数据的分类
        if (propertySet is None) or len(np.unique(x[:,propertySet], axis=0))==1:
            return self.__chooseLabel(y)
        #选取特征集合中最优的特征
        for i in propertySet:
            GainRatio, thres = self.__getGainRatio(x, y, i)
            if GainRatio > maxGainRatio:
                maxGainRatio = GainRatio
                threshold = thres
                bestProperty = i
        #用字典记录每一次的分类特征与信息增益率、阈值
        self.Gain_Ratio[self.LabelName[bestProperty]] = (maxGainRatio, threshold)
        #复制子特征集合
        subpropertySet = np.copy(propertySet).tolist()
        #如果选择预剪枝，则触发结束条件3：判断是否剪枝
        if self.mode == 'pre':
            ##（1）深度复制树结构
            afterTree = copy.deepcopy(self.treeDict)
            ##（2）判断是分类还是连续变量，分别追加叶节点
            leafDict[self.LabelName[bestProperty]] = dict()
            if threshold is None:
                for xvalue in np.unique(x[:,bestProperty]):
                    subx, suby = self.__splitDataset(x, y, bestProperty, xvalue)
                    leafDict[self.LabelName[bestProperty]][xvalue] = self.__chooseLabel(y) if len(suby)==0 else self.__chooseLabel(suby)
            else:
                for marker in ['less', 'more']:
                    subx, suby = self.__splitDataset(x, y, bestProperty, threshold, marker)
                    leafDict[self.LabelName[bestProperty]][marker+'/'+str(threshold)] = self.__chooseLabel(y) if len(suby) == 0 else self.__chooseLabel(suby)
            ##(3)将叶节点加到当前树中
            afterTree = leafDict if afterTree == dict() else self.__api(afterTree,treeIndex,leafDict)
            ##(4)判断是否剪枝，剪枝的话返回上一节点的类别，不剪枝的话替代当前树结构。
            pruningSign = self.__pruning(self.testX, self.testY, self.treeDict, afterTree)
            if pruningSign == 1:            
                return self.__chooseLabel(y)
            else:                           
                self.treeDict = afterTree
        #不符合结束条件和剪枝，可以继续递归
        treeIndex.append(self.LabelName[bestProperty])              #完善树的节点
        if threshold is None:
            subpropertySet.remove(bestProperty)                     #生成新属性集合
            for xvalue in np.unique(x[:,bestProperty]):
                treeIndex.append(xvalue)                            #添加树节点的属性值
                subx, suby = self.__splitDataset(x, y, bestProperty, xvalue)        #新数据的划分
                if len(suby) == 0:
                    return self.__chooseLabel(y)
                subTree = self.buildTree(subx, suby, subpropertySet, treeIndex)     #递归
                if self.LabelName[bestProperty] not in treeDict.keys():
                    treeDict[self.LabelName[bestProperty]] = dict()
                treeDict[self.LabelName[bestProperty]][xvalue] = subTree
                del treeIndex[-1]                                   #去除当前树节点的属性值
        else:
            for marker in ['less', 'more']:
                treeIndex.append(marker+'/'+str(threshold))
                subx, suby = self.__splitDataset(x, y, bestProperty, threshold, marker)
                subTree = self.buildTree(subx, suby, subpropertySet)
                if self.LabelName[bestProperty] not in treeDict.keys():
                    treeDict[self.LabelName[bestProperty]] = dict()
                treeDict[self.LabelName[bestProperty]][marker+'/'+str(threshold)] = subTree
                del treeIndex[-1]
        del treeIndex[-1]                                           #去除当前树的节点
        return treeDict

    #11、后剪枝
    def __afterpruning(self, x, y ,tree, treeIndex=[]):
        ##(1) 判断节点是否叶节点
        marker = 0
        node = list(tree.keys())[0]
        for nodeValue, leaf in tree[node].items():
            if isinstance(leaf, dict):
                marker += 1
        if marker == 0:             #是叶节点
            #复制剪枝前树
            preTree = copy.deepcopy(self.treeDict)
            #将新的树的叶子节点去掉
            preTree = self.__api(preTree,treeIndex,self.__chooseLabel(y),n=0)
            #判断是否需要剪枝
            pruningSign = self.__pruning(self.testX, self.testY, preTree, self.treeDict)
            if pruningSign==1:      #剪枝，返回节点分类
                self.treeDict = preTree
                return self.__chooseLabel(y)
            else:                   #不剪枝，返回原树结构
                return tree
        else:                       #是子节点，继续寻找叶节点
            treeIndex.append(node)
            for nodeValue, subtree in tree[node].items():
                #对于子节点进行递归
                if isinstance(subtree,dict):
                    treeIndex.append(nodeValue)
                    #判断是分类还是连续变量
                    marker, nodeValue = nodeValue.split('/') if isinstance(nodeValue, str) else None, nodeValue
                    subx, suby = self.__splitDataset(x, y, labels.index(node), float(nodeValue), marker)
                    tree[node][nodeValue] = self.__afterpruning(subx, suby, subtree, treeIndex)
                    del treeIndex[-1]       #删除节点属性值
            del treeIndex[-1]               #删除节点
        return tree

    #12、训练主函数
    def train(self, x, y, LabelName, testX, testY, mode=None):
        #初始特征集合
        self.trainSet = x
        self.label = y
        self.testX = testX
        self.testY = testY
        self.LabelName = LabelName
        self.mode = mode
        propertySet0 = list(range(np.shape(x)[1]))       #属性集合
        #递归函数
        self.treeDict = self.buildTree(x, y, propertySet0)
        #判断是否进行后剪枝
        if self.mode == 'after':
            treedict = self.__afterpruning(x, y, self.treeDict)
            self.treeDict = treedict

#区分训练集和测试集
trainx = Xdata[[0,1,2,5,6,9,13,14,15,16],:]
trainy = Ylabel[[0,1,2,5,6,9,13,14,15,16]]
testx = Xdata[[3,4,7,8,10,11,12],:]
testy = Ylabel[[3,4,7,8,10,11,12]]

#不剪枝
entropyTree = EntTree()
entropyTree.train(trainx, trainy, labels, testx, testy)
treeDict=entropyTree.treeDict  
        
#预剪枝
entropyTree = EntTree()
entropyTree.train(trainx, trainy, labels, testx, testy, mode='pre')
entropyTree.treeDict

#后剪枝
entropyTree = EntTree()
entropyTree.train(trainx, trainy, labels, testx, testy, mode='after')
treeDict = entropyTree.treeDict


#判断测试集准确率
for i in range(7):
    print(entropyTree._EntTree__predictTree(testx[i], treeDict))

    
from sklearn import tree      
clf = tree.DecisionTreeClassifier(criterion='entropy')      
clf.fit(x,y)        
clf.feature_importances_
print(tree.export_graphviz(clf,feature_names=labels,class_names=['0','1']) )
        

        
        
        
        
        
        
        
        
        
        
        
        
        




