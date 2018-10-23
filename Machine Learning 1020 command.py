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
        self.topN = 0                   #推荐前N个产品
        self.train = 0                  #训练集
        self.test = 0                   #测试集
        self.K = 0                      #选择相邻数
        self.W = 0                      #用户相似度
        self.rank = {}                  #推荐产品清单
        '''评测指标'''
        self.precision = 0              #准确率
        self.recall = 0                 #召回率
        self.cover = 0                  #覆盖率
        self.popular = 0                #流行度
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
    
    '''算法3：先转换成User-Item矩阵，再计算相似度'''
    def UserSimilarity3(self,ratings):
        #转换成U_I矩阵
        Items = list(set(ratings[:,1]))
        Users = list(set(ratings[:,0]))
        User_Item = np.zeros((len(Users),len(Items)))
        for i,j in ratings:
            U1 = Users.index(i)
            I1 = Items.index(j)
            User_Item[U1,I1] = 1
        #计算用户相似度，用矩阵形式
        w=np.ones((len(User_Item),len(User_Item)))
        for u in range(len(User_Item)):
            for v in range(len(User_Item)):
                if u ==v:
                    continue
                w[u][v] = np.dot(User_Item[u,:],User_Item[v,:])/np.sqrt((User_Item[u,:].sum()*User_Item[v,:].sum()))
        #计算用户相似度，用字典形式
        w=dict()
        for u in range(len(User_Item)):
            w[Users[u]] = dict()
            for v in range(len(User_Item)):
                if u==v:
                    continue
                w[Users[u]][Items[v]] = np.dot(User_Item[u,:],User_Item[v,:])/np.sqrt((User_Item[u,:].sum()*User_Item[v,:].sum()))
                
    '''产品推荐'''
    def command(self,train,User,W,K):
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
    '''计算评价指标'''
    def PandR(self,train,test,K,topN):
        NoUser = 0
        hit = 0                     #命中数量
        alltest = 0                 #测试集中的数量
        allcommand = 0              #所有推荐数量
        self.topN = topN
        self.K = K
        allItems = set()            #所有物品集合
        commandItems = set()        #所有推荐物品集合
        allPopular = dict()         #所有物品流行度
        avgPopular = 0              #推荐物品平均流行度
        '''先计算所有物品集合和流行度'''
        for u in train.keys():
            for item in train[u]:
                allItems.add(item)
                if item not in allPopular.keys():
                    allPopular[item] = 0
                allPopular[item] += 1
        '''正式计算四项评测指标：准确率、召回率、覆盖率、流行度'''
        for u in train.keys():
            try:
                testcommand = test[u]
                alltest += len(testcommand)
                rank = sorted(self.command(train,u,self.W,self.K).items(), key=operator.itemgetter(1), reverse=True)[:topN]
                for item, pui in rank:
                    if item in testcommand:
                        hit +=1
                    commandItems.add(item)
                    avgPopular += np.log(1+allPopular[item])
                    allcommand += 1
            except:
                NoUser += 1
        self.recall = hit/(alltest*1.0)
        self.precision = hit/(allcommand*1.0)
        print ('recall:{},precision:{}'.format(self.recall,self.precision))
        self.cover = len(commandItems)/len(allItems)
        self.popular = avgPopular/(len(testSet.keys())*topN)
        print ('cover:{},popular:{}'.format(self.cover,self.popular))

'''正式程序'''
Ucf = UserCF()
#区分数据集
Ucf.splitData(ratings,8,0,1234)
trainSet = Ucf.train
testSet = Ucf.test
#计算用户相似度和耗时(因为第一种算法保留了相似度为0的用户，第二种去除了，所以两者在选取推荐用户时排序问题有差异)
import time
start = time.clock()
Ucf.UserSimilarity2(trainSet)
end = time.clock()
print(end - start)
wi = Ucf.W
#计算推荐的准确率和召回率
Ucf.PandR(trainSet,testSet,80,10)
'''recall:0.12151931220325933,precision:0.25236508994004'''
'''cover:0.20010816657652786,popular:7.289074374222468'''













