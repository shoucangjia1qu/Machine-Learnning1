# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:27:40 2018

@author: ecupl
"""

###################决策树##################
'''1、了解算法ID3——信息增益'''
import numpy as np
'''计算分类别信息熵'''
b = 128+60+64+64+64+132+64+32+32      #购买
nb = 64+64+64+128+64                  #未购买
E_cate = -((b/(b+nb))*np.log2(b/(b+nb)) + (nb/(b+nb))*np.log2(nb/(b+nb)))   #分类信息熵
'''计算子节点信息熵'''
y = 64+64+128+64+64     #年轻类
yb = 64+64                  #年轻购买
ynb = 64+64+128             #年轻不购买
E_y = -((yb/(yb+ynb))*np.log2(yb/(yb+ynb)) + (ynb/(yb+ynb))*np.log2(ynb/(yb+ynb)))
e = 128+64+32+32        #中年类
eb = 128+64+32+32           #中年购买
E_e = -(eb/(eb+enb))*np.log2(eb/(eb+enb))
o = 60+64+64+132+64     #老年类
ob = 60+64+133              #年轻购买
onb = 64+63                 #年轻不购买
E_o = -((ob/(ob+onb))*np.log2(ob/(ob+onb)) + (onb/(ob+onb))*np.log2(onb/(ob+onb)))
'''计算信息增益'''
Py = y/(y+e+o)
Pe = e/(y+e+o)
Po = o/(y+e+o)
G_age = E_cate - (Py*E_y + Pe*E_e + Po*E_o)
'''G_age=0.2666969558634843'''

'''ID3算法实现'''
import numpy as np
import math, copy, pickle

'''定义类'''
class ID3Tree(object):
    '''1、初始化'''
    def __init__(self):         #构造方法
        self.tree={}            #生成树
        self.dataset=[]         #数据集
        self.labels=[]          #标签集
        
    '''2、数据导入函数'''
    def loadDataSet(self,path,labels):
        datalist=[]
        with open(path,"rb") as f:          #二进制形式读取文件
            content=f.read()
            rows = content.splitlines()     #分割文本，按行转换为一维表
            datalist=[row.split("\t") for row in rows if row.strip()]  #用制表符分割每个样本的变量值
            self.dataset = datalist
            self.labels = labels
    '''3、执行决策树函数'''
    def train(self):
        labels = copy.deepcopy(self.labels)     #深度复制lebels，相当于备份
        self.tree = self.buildTree(self,dataset,labels)

    '''4、创建决策树主程序'''
    def buildTree(self,dataset,labels):
        catelist=[data[-1] for data in dataset]     #抽取数据源标签列
        '''程序终止，只有一种分类标签'''
        if catelist.count(catelist[0]) == len(catelist):
            return catelist[0]
        '''只有一个变量，无法再分'''
        if len(dataset[0])==1:
            return self.maxCate(catelist)
        '''算法核心：返回最优特征轴'''
        bestfeat = self.getBestFeat(self,dataset)
        bestfeatlabel = labels[bestfeat]
        tree={bestfeatlabel:()}
        del labels[bestfeat]
        '''抽取最优特征轴的列向量'''
        uniqueVals = set([data[bestfeat] for data in dataset])      #特征轴的值
        for value in uniqueVals:
            sublabels=label[:]
            splitdata = self.splitdateset(dateset,bestfeat,value)
            subTree = self.buildTree(splitdata,sublabels)
            tree[bestfeatlabel][value]=subTree
        return tree
    
    '''5、计算出现次数最多的类别标签'''
    def maxCate(self,catelist):
        items = dict([(i, catelist.count(i),) for i in catelist])    
        maxc = list(items.keys())[list(items.values()).count(max(list(items.values())))]
        return maxc

    '''6、计算最优特征'''
    def getBestFeat(self,dataset):
        numFeatures = len(dataset[0]-1)     #计算特征维
        baseEntropy = self.computeEntropy(dataset)      #计算信息熵，基础的
        bestgain=0          #初始化信息增益
        bestFeature = -1    #初始化最优特征轴
        for x in range(numFeatures):
            uniqueVals = set([data[x] for data in dataset])
            newEntropy=0
            for value in uniqueVals:
                subdataset = self.splitdataset(dataset,i,value)     #切分数据集，取出需要计算的部分
                pro = len(subdataset)/len(dataset)
                newEntropy += pro*self.computeEntropy(subdataset)
            gain = baseEntropy - newEntropy         #计算最大增益
            if gain>bestgain:
                bestgain = gain
                bestFeature = x
        return  bestFeature
    
    '''7、计算信息熵'''
    def computeEntropy(self.dataset):
        cates = [i for i in dataset[-1]]
        datalen = len(dataset)
        items = dict([(cate, cates.count(cate)) for cate in cates])
        for key in items.keys():
            pro = float(items[key])/datalen
            Entropy = -pro*np.log2(pro)
        return Entropy
    
    '''8、划分数据集'''
    def splitdateset(self,dataset,axis,value):
        rtnlist=[]
        for data in dataset:
            if data[axis]==value:
                rtndata=data[:axis]
                rtndata.extend(data[axis+1:])
                rtnlist.append(rtndata)
        return rtnlist

'''训练树'''










