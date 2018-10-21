# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:21:33 2018

@author: ecupl
"""
#####################推荐算法#######################
import numpy as np
from numpy import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import operator
import os

os.chdir(r'D:\mywork\test\command')
curpath = os.getcwd()
#with open("users.txt","rb") as f:
#    users = f.readlines
'''读入数据'''
users = pd.read_table('users.dat',sep = '::',header=None,engine='python')
movies = pd.read_table('movies.dat',sep = '::',header=None,engine='python')
ratings = pd.read_table('ratings.dat',sep = '::',header=None,engine='python')
'''修改列名'''
users.columns=['UserID','Gender','Age','Occupation','Zip-code']
movies.rename(columns={0:'MovieID',1:'Title',2:'Genres'}, inplace = True)
ratings.rename(columns={0:'UserID',1:'MovieID',2:'Rating',3:'Timestamp'}, inplace=True)
ratings = np.array(ratings.iloc[:,0:2])
'''切分数据集,交叉验证啊'''
def splitData(data, M, k,seed):
    test = []
    train = []
    np.random.seed(seed)
    for user, item in data:
        if np.random.randint(0,M) == k:
            test.append([user,item])
        else:
            train.append([user,item])
    return train,test
train, test = splitData(ratings,8,0,1234)
trainSet = dict()       #转换成字典，一个用户对应很多电影评分
for i in train :
    if i[0] not in trainSet.keys():
        trainSet[i[0]] = [i[1]]
    else:
        trainSet[i[0]].append(i[1])

'''基于用户的推荐算法1：夹角余弦计算用户对产品的相似度'''
#两两用户的评分电影的交集/用户推荐产品的长度乘积,储存为array
def UserSimilarity(train):
    w=np.ones((len(train.keys()),len(train.keys())))
    i = 0
    for u in train.keys():
        j = 0
        for v in train.keys():
            if u != v:
                w[i,j] = len(set(train[u])&set(train[v]))
                w[i,j] /= np.sqrt(len(set(train[u]))*len(set(train[v])))
            j+=1
        i+=1           
    return w
W=UserSimilarity(trainSet)
#两两用户的评分电影的交集/用户推荐产品的长度乘积,储存为字典
def UserSimilarity(train):
    w=dict()
    for u in train.keys():
        w[u] = dict()
        for v in train.keys():
            if u == v:
                continue
            w[u][v] = len(set(train[u])&set(train[v]))/np.sqrt(len(set(train[u]))*len(set(train[v])))
    return w
W1=UserSimilarity(trainSet)

'''基于用户的推荐算法2：夹角余弦计算用户对产品的相似度'''
#先转换成物品的客户有哪些
def Item_Users(train):
    #将评过同一部电影的用户放在一起
    Item_Users = dict()
    for user,items in trainSet.items():
        for i in items:
            if i not in Item_Users.keys():
                Item_Users[i] = set()
            Item_Users[i].add(user)
    #提取物品相关的客户数，以及每个用户的推荐数量
    N = dict()
    Ur = dict()
    for items,users in Item_Users.items():
        for u in users:
            if u not in N.keys():
                N[u] = 0
            N[u] += 1
            if u not in Ur:
                Ur[u] = dict()
            for subu in users:
                if u==subu:
                    continue
                if subu not in Ur[u]:
                    Ur[u][subu] = 0
                Ur[u][subu] += 1
    W=dict()
    for u,subu in Ur.items():
        W[u] = dict()
        for u1,times in subu.items():
            W[u][u1] = times/np.sqrt(N[u]*N[u1])
            
W = Item_Users(trainSet)

'''对产品推荐的偏好'''
def command(dataSet,U,W,K):
    rank = dict()
    having_items = dataSet[U]
    for u , wi in sorted(W[U].items(), key=operator.itemgetter(1), reverse=True)[:K]:
        for item in trainSet[u]:
            if item not in having_items:
                if item not in rank.keys():
                    rank[item] = 0
                rank[item] += wi*1.0
    return rank
        



















