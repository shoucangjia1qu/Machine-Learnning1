# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:45:36 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
os.chdir(r"D:\mywork\test")

#Constrained_K-means聚类(约束k均值聚类)，一种半监督学习算法
class Ckmeans(object):
    #1、属性
    def __init__(self):
        self.K = 0          #聚类簇的个数
        self.CntPoint = 0   #聚类中心点
        self.labels = 0     #聚类类别
        self.dists = 0      #每个点距离中心点的距离
        self.M_Link = 0     #必连集合
        self.C_Link = 0     #勿连集合
    
    #2、距离公式(采用欧式距离)
    def calDist(self,v0,v1):
        return np.linalg.norm(v0-v1)
    
    #3、中心标准化原数据
    def stdData(self,train):
        stdtrain = (train-np.mean(train,axis=0))/np.std(train,axis=0)
        return stdtrain
    
    #4、初始化中心点
    def firstCntPoint(self,train,K):
        m,n=np.shape(train)
        ratio = np.random.random((K,n))
        minvalue = np.min(train,axis=0)
        maxvalue = np.max(train,axis=0)
        points = minvalue+ratio*(maxvalue-minvalue)
        #本次测试固定
        points = train[[5,11,26],:]
        return points
    
    #5、判断是否符合越是条件，即必连和勿连条件
    def is_contrained(self, xi_idx, xi_label, clusters, M, C):
        """
        检查样本的类别是否符合约束条件
        Input:
            xi_idx:需要检查的样本下标
            xi_label:样本的簇类
            clusters:所有样本的簇类
            M:必连集合
            C:勿连集合
        """
        #先检查必连，遍历必连中的集合组合
        mValue = False
        for Set in M:
            #如果已经判定目标不符合必连条件，则结束后面的遍历
            if mValue == True:
                break
            #如果样本在该集合组合中
            if xi_idx in Set:
                #逐个检验集合中的其他样本是否和目标样本同簇
                for ss in Set:
                    #目标样本跳过
                    if ss == xi_idx:
                        continue
                    #其他样本的簇
                    ss_label = clusters[ss]
                    if (ss_label==xi_label) or (ss_label==-1):
                        #簇和目标样本相同或者为初始值，则没毛病
                        continue
                    else:
                        #否则，说明目标样本在这个簇中不符合约束条件，返回True，还要继续迭代
                        mValue = True
                        break
            else:
                pass
        #再检查勿连
        cValue = False
        for Set in C:
            if cValue == True:
                break
            if xi_idx in Set:
                for ss in Set:
                    if ss == xi_idx:
                        continue
                    ss_label = clusters[ss]
                    if ss_label != xi_label:
                        continue
                    else:
                        cValue = True
                        break
            else:
                pass
        #返回判定值
        return mValue, cValue

    
    #6、主循环
    def cluster(self, train, K, M, C, std=False):
        """
        Input:
            train:训练集数
            K:聚类个数
            M:必连集合
            C:勿连集合
            std:是否中心标准化，默认否
        """
        #4-1 判断数据是否需要标准化
        if std==True:
            train = self.stdData(train)
        #4-2 初始化中心点
        CntPoint = self.firstCntPoint(train,K)
        #4－3 开始聚类
        m, n = np.shape(train)
        labellist = np.zeros(m)-1               #令所有样本的簇的初始值为-1
        distlist = np.zeros(m)
        flag = True
        while flag:
            flag = False
            #每个点的聚类标签循环一遍
            for i in range(m):
                dists = [self.calDist(train[i,:],CntPoint[c,:]) for c in range(K)]
                #每个点都要检查一遍是否符合约束条件
                subdists = copy.deepcopy(dists)
                while True:
                    mindist = min(subdists)
                    minIdx = dists.index(mindist)
                    mValue, cValue = self.is_contrained(i, minIdx, labellist, M, C)
                    #当必连和勿连中有一个不符合要求时，就得重新迭代
                    if (mValue==False) and (cValue==False):
                        break
                    else:
                        subdists.remove(mindist)
                        #如果列表已经为空了，则没有适合该样本的分类，报错，并跳出执行
                        if len(dists) == 0:
                            print("第%d个没有符合的簇"%i)
                            break
                #只要有一个样本的类不一致，就要再次循环
                if minIdx != labellist[i]:
                    flag = True
                labellist[i] = minIdx
                distlist[i] = mindist
            #迭代每个中心点
            for j in range(K):
                dataSet = train[np.nonzero(labellist==j)[0],:]
                if len(dataSet) != 0:
                    CntPoint[j,:] = np.mean(dataSet,axis=0)
                else:
                    print("第{}个点无近似样本".format(j))
            
            #每次迭代画个图
            #1、簇内的图
            for clu in range(K):
                plt.scatter(CntPoint[clu,0],CntPoint[clu,1],c="r",marker="D",linewidths=5)
                plt.scatter(train[np.nonzero(labellist==clu)[0],0], train[np.nonzero(labellist==clu)[0],1])
            #2、原始簇的图
            plt.scatter(train[np.nonzero(labellist==-1)[0],0], train[np.nonzero(labellist==-1)[0],1],marker=">",linewidths=3,c="y")
            #3、画约束条件
            for Set in M:
                ss = list(Set)
                plt.plot(train[ss,0], train[ss,1], "r-")
            for Set in C:
                ss = list(Set)
                plt.plot(train[ss,0], train[ss,1], "b--")
            plt.show()
        self.K = K
        self.CntPoint = CntPoint
        self.labels = labellist
        self.dists = distlist
        self.M_Link = M
        self.C_Link = C






#开始训练
if __name__ == "__main__":
    ##############西瓜集数据4.0
    data = np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],[0.403,0.237],[0.481,0.149],
                     [0.437,0.211],[0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],
                     [0.360,0.370],[0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],[0.748,0.232],
                     [0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],[0.751,0.489],[0.532,0.472],[0.473,0.376],
                     [0.725,0.445],[0.446,0.459]])
    M = [{3,24},{11,19},{13,16}]
    C = [{1,20},{12,22},{18,22}]
    K = 3
    Ckm = Ckmeans()
    Ckm.cluster(data, K, M, C)
    clusters = Ckm.labels
