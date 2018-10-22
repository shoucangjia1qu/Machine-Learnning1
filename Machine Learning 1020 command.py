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

class UserCF(object):
    '''定义属性'''
    def __init__(self):
        self.train = 0
        self.test = 0
        self.K = 0
        self.W = 0
        self.rank = {}
        self.precision = 0
        self.recall = 0
        self.cover = 0
        self.popular = 0
    '''切分数据集，以方便交叉验证'''
    def splitData(self,data,M,m0,seed):
        tr = {}
        te = {}
        np.random.seed(seed)        #使用随机种子
        for user, item in data:
            if random.randint(0,M) == m0:
                if user not in te.keys():
                    te[user] = set()
                te[user].add(item)
            else:
                if user not in tr.keys():
                    tr[user] = set()
                tr[user].add(item)
        self.train = tr
        self.test = te
    
    '''算法1：直接求夹角余弦计算用户相似度'''
    def UserSimilarity1(self,train):
        w=dict()
        for user in train.keys():
            w[user] = dict()
            for subuser in train.keys():
                if user==subuser:
                    continue
                w[user][subuser] = len(train[user]&train[subuser])/np.sqrt(len(train[user])*len(train[subuser]))
    self.W = w

    '''算法2：先转换为Item-Users形式，再计算User-User之间的相似度'''
    def UserSimilarity2(self,train):
        #转换为I_U形式
        item_users = dict()
        for user, items in train.items():
            for item in items:
                if item not in item_users.keys():
                    item_users[item] = set()
                item_users[item].add(user)
        #计算U_U相似度
        U_Itimes = dict()       #用户对应产品的数量
        for user in train.keys():
            U_Itimes[user] = len(train[user])
        U_Utimes = dict()       #用户与用户之间交集个数
        for item, users in item_users.items():
            for u in users:
                if u not in U_Utimes:
                    U_Utimes[u] = dict()
                for subuser in users:
                    if u==subuser:
                        continue
                    if subuser not in U_Utimes[u].keys():
                        U_Utimes[u][subuser] = 0
                    U_Utimes[u][subuser] += 1
        w=dict()
        for user,item in U_Utimes.items():
            w[user] = dict()
            for subuser,times in item.items():
                w[user][subuser] = times/np.sqrt(U_Itimes[user]*U_Itimes[subuser])
        self.W = w
    
    '''产品推荐'''
    def command(self,train,User,W,K):
        self.K = K
        rank = dict()
        having_items = train[User]
        for u, wi in sorted(W[User].items(), key=operator.itemgetter(1), reverse=True)[:K]:
            for item in train[u]:
                if item in having_items:
                    continue
                if item not in rank.keys():
                    rank[item] = 0
                rank[item] += wi*1.0
        return rank
    '''计算召回率'''
    def Recall(self,train,test,K):
        hitcommand = 0
        allcommand = 0
        for u in train.keys():
            testcommand = test[u]
            allcommand += len(testcommand)
            rank = self.command(train,u,self.W,K)
            for item, pui in rank.items():
                if item in testcommand:
                    hitcommand +=1
        self.recall = hitcommand/(allcommand*1.0)
        print (hitcommand/(allcommand*1.0))
    
                
                    
    



















