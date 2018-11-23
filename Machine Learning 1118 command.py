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
'''1-3推荐'''
def command(train,user,W,K,t0,beta):
    rank = dict()
    I_Tdict = train[user]
    for i,tui in I_Tdict.items():
        for j,wij in sorted(W[i].items(), key=operator.itemgetter(1), reverse=True)[:K]:
            if j in I_Tdict.keys():
                continue
            if j not in rank.keys():
                rank[j] = 0
            rank[j] += 1.0*wij/(1+beta*(t0-tui))
#用户的行为i发生的时间越近，其相似的物品j的权重越高

'''2、User_CF'''
'''2-1生成item-user矩阵'''
IUSet = dict()
for u,i_t in trainSet.items():
    for i,t in i_t.items():
        if i not in IUSet.keys():
            IUSet[i] = dict()
        if u not in IUSet[i].keys():
            IUSet[i][u] = t
'''2-2生成用户相似度矩阵'''
def UserSim(train,alpha):
    Sim = dict()
    UserNum = dict()
    for i,u_t in train.items():
        for ui,ti in u_t.items():
            if ui not in UserNum.keys():
                UserNum[ui] = 0
            UserNum[ui] += 1
            if ui not in Sim.keys():
                Sim[ui] = dict()
            for uj,tj in u_t.items():
                if ui==uj:
                    continue
                if uj not in Sim[ui].keys():
                    Sim[ui][uj] = 0
                Sim[ui][uj] += 1/(1+alpha*abs(ti-tj))
    #计算用户之间的相似度
    W = dict()
    for ui,uj_wij in Sim.items():
        W[ui] = dict()
        for uj,Wij in uj_wij.items():
            W[ui][uj] = Wij/np.sqrt(UserNum[ui]*UserNum[uj])
    return W
Users_W = UserSim(IUSet,0.8)
'''2-3推荐'''
def command(train,user,W,K,t0,beta):
    rank = dict()
    I_Tdict = train[user]
    for u,w in sorted(W[user].items(), key=operator.itemgetter(1), reverse=True)[:K]:
        for i,tui in train[u].items():
            if i in I_Tdict.keys():
                continue
            if i not in rank.keys():
                rank[i] = w/(1+beta*(t0-tui))
    return rank
rank = command(trainSet,1,Users_W,10,978300770,0.8)


######################################
#                                    #
#  基于Movielens数据的各类算法大比拼   #
#                                    #
######################################
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
'''0、读入数据'''
ratings = pd.read_table('ratings.dat',sep = '::',header=None,engine='python')
'''0-1修改列名，划分数据集(每个用户最近日期的item作为测试集)'''
ratings.rename(columns={0:'UserID',1:'MovieID',2:'Rating',3:'Timestamp'}, inplace=True)
UserList = list(set(ratings.UserID))            #用户列表
ItemList = list(set(ratings.MovieID))           #物品列表
TimeList = list(set(ratings.Timestamp))         #时间戳列表
trainSet = pd.DataFrame(columns=['UserID','MovieID','Rating','Timestamp'])
testSet = pd.DataFrame(columns=['UserID','MovieID','Rating','Timestamp'])
for u in UserList:
    tempSet = ratings[ratings.UserID==u]
    tempSet.sort_values("Timestamp",inplace=True)
    trainSet = trainSet.append(tempSet.iloc[:-1,:],ignore_index=True)
    testSet = testSet.append(tempSet.iloc[-1,:],ignore_index=True)
'''0-2通用数据。计算训练集的UI矩阵和UIt矩阵,UIt和IUt字典'''
U_IMatrix = np.zeros((len(UserList),len(ItemList)))
U_I_TMatrix = np.zeros((len(UserList),len(ItemList)))
for u,i,s,t in np.array(trainSet):
    uIdx = UserList.index(u)
    iIdx = ItemList.index(i)
    U_IMatrix[uIdx,iIdx] = 1
    U_I_TMatrix[uIdx,iIdx] = t
U_IDict = dict()
I_UDict = dict()
for u,i,s,t in np.array(trainSet):
    if u not in U_IDict.keys():
        U_IDict[u] = dict()
    if i not in U_IDict[u].keys():
        U_IDict[u][i] = 0
    U_IDict[u][i] = t
    if i not in I_UDict.keys():
        I_UDict[i] = dict()
    if u not in I_UDict[i].keys():
        I_UDict[i][u] = 0
    I_UDict[i][u] = t
'''0-3通用函数。夹角余弦函数'''
def cosdist(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
'''1、ICF算法，且考虑进用户活跃度'''
'''1-1计算用户相似度，矩阵运算'''
def cosdist_ICF(Wu,v1,v2):
    return np.dot(Wu,np.multiply(v1,v2))/(np.linalg.norm(v1)*np.linalg.norm(v2))
def ICF_W(train):
    row,col = train.shape
    '''计算用户活跃度数组'''
    Wu = np.zeros(row)
    for u in range(row):
        Wu[u] = 1/np.log(1+len(train[u,:][train[u,:]>0]))
    '''计算物品相似度'''
    W = np.ones((col,col))
    for i in range(col):
        for j in range(col):
            if i==j:
                continue
            W[i,j] = cosdist_ICF(Wu,train[:,i],train[:,j])
    return W
ICF_WMa = ICF_W(U_IMatrix)

'''1-1计算物品相似度，字典运算（用于和矩阵运算进行核对与比较）'''
'''一致'''
def ICF_WD(train):
    #计算物品之间重合系数和单个物品的用户数
    Sim = dict()
    ItemNum = dict()
    for u,i_t in train.items():
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
                Sim[i][j] += 1/np.log(1+len(i_t))
    #计算物品之间的相似度
    W = dict()
    for i,wj in Sim.items():
        W[i] = dict()
        for j,wij in wj.items():
            W[i][j] = wij/np.sqrt(ItemNum[i]*ItemNum[j])
    return W

'''2、UCF算法，且考虑进物品流行度的因素'''
'''2-1定义专属距离公式'''
def cosdist_UCF(Wi,v1,v2):
    return np.dot(Wi,np.multiply(v1,v2))/(np.linalg.norm(v1)*np.linalg.norm(v2))
'''2-2定义用户相似度矩阵，训练集为UI矩阵'''
def UCF_W(train):
    row,col = train.shape
    '''计算物品流行度数组'''
    Wi = np.zeros(col)
    for i in range(col):
        Wi[i] = 1/np.log(1+len(train[:,i][train[:,i]>0])+1.0e-6)                #物品2039无训练集，所以加了个exp
    '''计算用户相似度'''
    W = np.ones((row,row))
    for u in range(row):
        for v in range(row):
            if u==v:
                continue
            W[u,v] = cosdist_UCF(Wi,train[u,:],train[v,:])
        print("{}已完成".format(u))
    return W
UCF_WMa = UCF_W(U_IMatrix)

'''3、基于图的算法，随机游走'''
def PersonalRank(train,Uroot,alpha,Iters):
    global UserList,ItemList,TimeList
    row,col = train.shape
    '''建立用户排序和物品排序'''
    uRank = np.zeros((1,row))
    iRank = np.zeros((1,col))
    uRank[0,UserList.index(Uroot)] = 1
    '''根据用户和物品节点的出度，计算权重'''
    IRatio_U = np.zeros((row,col))
    URatio_I = np.zeros((row,col))
    for iIdx in range(col):
        IRatio_U[:,iIdx] = train[:,iIdx]/train.sum(axis=1)
    for uIdx in range(row):
        URatio_I[uIdx,:] = train[uIdx,:]/(train.sum(axis=0)+1.0e-6)
    '''迭代每个节点的概率'''
    for step in range(Iters):
        itemp = np.dot(alpha*uRank,IRatio_U)
        utemp = np.dot(alpha*iRank,URatio_I.T)
        utemp[0,UserList.index(Uroot)] += (1-alpha)
        iRank = itemp
        uRank = utemp
        print("第{}次迭代".format(step))
    return iRank,uRank
iRank,uRank = PersonalRank(U_IMatrix,1,0.8,10)

'''3、基于图的算法，随机游走，字典运算和上面进行比较'''
'''3-1创建用户和物品节点'''
G = dict()
for u,i,s,t in np.array(trainSet):
    uName = 'u{}'.format(u)
    iName = 'i{}'.format(i)
    '''加入用户节点'''
    if uName not in G.keys():
        G[uName] = dict()
    if i not in G[uName].keys():
        G[uName][iName] = 0
    G[uName][iName] = t
    '''加入物品节点'''
    if iName not in G.keys():
        G[iName] = dict()
    if u not in G[iName].keys():
        G[iName][uName] = 0
    G[iName][uName] = t
'''3-2随机游走'''
def PersonalRank(G,alpha,root,Maxiter):
    rank = dict()
    rank = {x:0 for x in G.keys()}
    rank[root] = 1
    for i in range(Maxiter):
        temp = {x:0 for x in G.keys()}
        for leaf, subpath in G.items():
            for subleaf, path in subpath.items():
                temp[subleaf] += alpha*rank[leaf]/(1.0*len(subpath))
        temp[root] += 1-alpha
        rank = temp
    return rank
GRank = PersonalRank(G,0.8,'u1',10)







