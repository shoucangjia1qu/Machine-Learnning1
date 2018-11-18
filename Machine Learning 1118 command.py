# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:21:33 2018

@author: ecupl
"""
#####################推荐算法#######################
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import operator
import os

os.chdir(r'D:\mywork\test\command\UI_CF')
curpath = os.getcwd()
#with open("users.txt","rb") as f:
#    users = f.readlines
'''读入数据'''
'''Item_CF'''
ratings = pd.read_table('ratings.dat',sep = '::',header=None,engine='python')
'''修改列名'''
ratings.rename(columns={0:'UserID',1:'MovieID',2:'Rating',3:'Timestamp'}, inplace=True)
UIwithTime = ratings.drop('Rating',axis=1)
dataSet = np.array(UIwithTime)
######################################
#                                    #
#  时间上下文信息(协同过滤算法的改进)  #
#                                    #
######################################
'''1、Item_CF'''
'''1-1、拆解数据集'''
def splitData(data,M,m0,seed):
        tr = {}
        te = {}
        np.random.seed(seed)        #使用随机种子
        for u,i,t in data:
            if np.random.randint(0,M) == m0:
                if u not in te.keys():
                    te[u] = dict()
                te[u][i] = t
            else:
                if u not in tr.keys():
                    tr[u] = dict()
                tr[u][i] = t
        train = tr
        test = te
        return train,test
trainSet,testSet = splitData(dataSet,8,0,1234)
'''1-2计算物品相似度'''
def ItemSim(dictData,alpha):
    #计算物品之间重合系数和单个物品的用户数
    Sim = dict()
    ItemNum = dict()
    for u,i_t in dictData.items():
        for i,Ti in i_t.items():
            if i not in ItemNum.keys():
                ItemNum[i] = 0
            ItemNum[i] += 1
            if i not in Sim.keys():
                Sim[i] = dict()
            for j,Tj in i_t.items():
                if i==j:
                    continue
                if j not in Sim[i].keys():
                    Sim[i][j] = 0
                Sim[i][j] += 1/(1+alpha*abs(Ti-Tj))
    #计算物品之间的相似度
    W = dict()
    for i,wj in Sim.items():
        W[i] = dict()
        for j,wij in wj.items():
            W[i][j] = wij/np.sqrt(ItemNum[i]*ItemNum[j])
    return W

Items_W = ItemSim(trainSet,0.8)







