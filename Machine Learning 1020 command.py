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
        for u in test.keys():
            testcommand = test[u]
            alltest += len(testcommand)
            rank = sorted(self.command(train,u,self.W,self.K).items(), key=operator.itemgetter(1), reverse=True)[:topN]
            for item, pui in rank:
                if item in testcommand:
                    hit +=1
                commandItems.add(item)
                avgPopular += np.log(1+allPopular[item])
                allcommand += 1
        self.recall = hit/(alltest*1.0)
        self.precision = hit/(allcommand*1.0)
        print ('recall:{},precision:{}'.format(self.recall,self.precision))
        self.cover = len(commandItems)/len(allItems)
        self.popular = avgPopular/(len(testSet.keys())*topN)
        print ('cover:{},popular:{}'.format(self.cover,self.popular))
    #####################################################################
    '''改进后算法，UserCF-IIF'''
    def UserSimilarityIIF(self,train):
        item_user = dict()
        for user, items in train.items():
            for item in items:
                if item not in item_user.keys():
                    item_user[item] = set()
                item_user[item].add(user)
        
        U_Itimes = dict()
        for user,items in train.items():
            U_Itimes[user] = len(items)
        U_Utimes = dict()
        for item, users in item_user.items():
            for u in users:
                if u not in U_Utimes.keys():
                    U_Utimes[u] = dict()
                for subu in users:
                    if u==subu:
                        continue
                    if subu not in U_Utimes[u].keys():
                        U_Utimes[u][subu] = 0
                    U_Utimes[u][subu] += 1/np.log(1+len(users))
        
        W = dict()
        for u, subus in U_Utimes.items():
            W[u] = dict()
            for v,times in subus.items():
                W[u][v] = times/np.sqrt(U_Itimes[u]*U_Itimes[v])
        self.W = W
                

'''正式程序'''
Ucf = UserCF()
#区分数据集
Ucf.splitData(ratings,8,0,1234)
trainSet = Ucf.train
testSet = Ucf.test
###要点###计算用户相似度和耗时(因为第一种算法保留了相似度为0的用户，第二种去除了，所以两者在选取推荐用户时排序问题有差异)
import time
start = time.clock()
Ucf.UserSimilarity2(trainSet)
end = time.clock()
print(end - start)
wi = Ucf.W

Ucf.PandR(trainSet,testSet,80,10)
'''recall:0.12151931220325933,precision:0.25236508994004'''
'''cover:0.20010816657652786,popular:7.289074374222468'''

#计算用户相似度UserCF_IIF
import time
start = time.clock()
Ucf.UserSimilarityIIF(trainSet)
end = time.clock()
print(end - start)
wiif = Ucf.W
#计算推荐的准确率和召回率
'''recall:0.12193635313743102,precision:0.2532311792138574'''
'''cover:0.20930232558139536,popular:7.25838399476049'''

######################################
#                                    #
#            ItemCF算法              #
#                                    #
######################################

'''计算相似度'''
def ItemSimilarity(train):
    I_times = dict()            #统计每个物品出现的次数
    I_Itimes = dict()           #统计物品与物品之间被几个用户喜欢
    for user, items in train.items():
        for item in items:
            if item not in I_Itimes.keys():
                I_Itimes[item] = dict()
                I_times[item] = 0
            I_times[item] += 1
            for subitem in items:
                if item == subitem:
                    continue
                if subitem not in I_Itimes[item].keys():
                    I_Itimes[item][subitem] = 0
                I_Itimes[item][subitem] += 1
    
    W = dict()                  #物品之间的相似度
    for item, others in I_Itimes.items():
        W[item] = dict()
        for subitem, times in others.items():
            W[item][subitem] = times/np.sqrt(I_times[item]*I_times[subitem])

'''基于物品相似度的推荐'''                
def command(train,U,W,K):
    rank = dict()
    havingItems = train[U]
    for item in havingItems:
        for commandItem, wi in sorted(W[item].items(),key=operator.itemgetter(1), reverse=True)[:K]:
            if commandItem in havingItems:
                continue
            if commandItem not in rank.keys():
                rank[commandItem] = 0
            rank[commandItem] += 1*wi
    return rank

'''基于物品相似度的推荐，加上解释度'''                
def command(train,U,W,K):
    rank = dict()
    reason = dict()
    havingItems = train[U]
    for item in havingItems:
        for commandItem, wi in sorted(W[item].items(),key=operator.itemgetter(1), reverse=True)[:K]:
            if commandItem in havingItems:
                continue
            if commandItem not in rank.keys():
                rank[commandItem] = 0
                reason[commandItem] = dict()
            rank[commandItem] += 1*wi
            #加入推荐物品的解释理由
            if item not in reason[commandItem].keys():
                reason[commandItem][item] = 0
            reason[commandItem][item] += 1*wi
    return rank,reason

'''离线实验评价指标：准确率、召回率、覆盖度、流行度'''
def PandR(train,test,K,topN):
    hit = 0             #命中数量
    allcommand = 0      #总推荐数
    alltest = 0         #测试集总推荐数  
    allitems = set()    #所有物品集合
    allpopular = dict()   #所有物品流行度
    cmditems = set()    #推荐物品的集合
    cmdpopular = 0      #推荐物品的总流行度
    
    #计算所有物品的数量和流行度
    for user,items in train.items():
        for i in items:
            allitems.add(i)
            if i not in allpopular.keys():
                allpopular[i] = 0
            allpopular[i] += 1
    #准确率和召回率
    for U,items in test.items():
        rank = sorted(command(train,U,W,K).items(), key=operator.itemgetter(1),reverse=True)[:topN]     #根据物品相似度推荐的所有物品
        alltest += len(items)           #测试集所有实际物品
        for commanditem, score in rank:
            if commanditem in items:
                hit += 1
            allcommand += 1
            cmditems.add(commanditem)
            cmdpopular += np.log(1 + allpopular[commanditem])
    precision = hit/allcommand
    recall = hit/alltest
    cover = len(cmditems)/len(allitems)
    popular = cmdpopular/allcommand
    print ('recall:{},precision:{}'.format(recall,precision))
    print ('cover:{},popular:{}'.format(cover,popular))
'''recall:0.10586423713589119,precision:0.21985343104596936'''
'''cover:0.19118442401298,popular:7.2502167984452'''
   
'''算法改进1：基于用户活跃度的惩罚，对高活跃度的用户其物品之间的相似度贡献变小'''    
def ItemSimilarityIUF(train):
    I_times = dict()            #统计每个物品出现的次数
    I_Itimes = dict()           #统计物品与物品之间被几个用户喜欢
    for user, items in train.items():
        for item in items:
            if item not in I_Itimes.keys():
                I_Itimes[item] = dict()
                I_times[item] = 0
            I_times[item] += 1
            for subitem in items:
                if item == subitem:
                    continue
                if subitem not in I_Itimes[item].keys():
                    I_Itimes[item][subitem] = 0
                I_Itimes[item][subitem] += 1/np.log(1+len(items))
    
    W = dict()                  #物品之间的相似度
    for item, others in I_Itimes.items():
        W[item] = dict()
        for subitem, times in others.items():
            W[item][subitem] = times/np.sqrt(I_times[item]*I_times[subitem])
'''recall:0.10821410239958938,precision:0.22473351099267155'''
'''cover:0.1755002704164413,popular:7.350681913999984'''

'''算法改进2：物品相似度的归一化(有点问题,究竟用Wj还是Wi)'''
Wmax = dict()
#横向取最大值和纵向取最大值是一样的，所以这里是横向取了最大值
for i,i_items in W.items():
    Wmax[i] = max(list(i_items.values()))
Wj = copy.deepcopy(W)
for i,i_items in Wj.items():
    for j,v in i_items.items():
        Wj[i][j] = v/Wmax[j]
#Wj
'''recall:0.038415886051584754,precision:0.07978014656895403'''
'''cover:0.39724175229853975,popular:6.026280574239222'''

Wi = copy.deepcopy(W)
for i,i_items in Wi.items():
    for j,v in i_items.items():
        Wi[i][j] = v/Wmax[i]
#Wi
'''recall:0.11200757089695881,precision:0.2326115922718188'''
'''cover:0.22363439697133586,popular:7.242104855359775'''


