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
    return rank
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
                rank[i] = 0
            rank[i] += w/(1+beta*abs(t0-tui))
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
import copy

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
    return np.dot(Wu,np.multiply(v1,v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)+1.0e-6)
def ICF_W(train):
    row,col = train.shape
    '''计算用户活跃度数组'''
    Wu = np.zeros(row)
    for u in range(row):
        Wu[u] = 1/np.log(1+len(train[u,:][train[u,:]>0]))
    '''计算物品相似度'''
    W = np.zeros((col,col))
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
    W = np.zeros((row,row))
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

'''4、按时间流行度推荐，用矩阵运算'''
def RecentPop(train,alpha,T):
    row,col = train.shape
    PopMa=np.zeros(col)
    for i in range(col):
        PopMa[i] = 0
        temp = train[:,i][train[:,i]>0]
        PopMa[i] = np.sum(1/(1+alpha*(T-temp)))
    return PopMa
PopMa = RecentPop(U_I_TMatrix,0.8,max(TimeList))      

'''4、按时间流行度推荐，用字典运算'''
def RecentPop_Dict(train,alpha,T):
    PopDict = dict()
    for u,i,s,t in np.array(trainSet):
        if i not in PopDict.keys():
            PopDict[i] = 0
        PopDict[i] += 1/(1+alpha*(T-t))
    return PopDict
PopDict = RecentPop_Dict(trainSet,0.8,max(TimeList))

'''5、TICF算法，考虑用户对物品的时间戳'''
def TICF_W(train,alpha):
    OneMa = copy.deepcopy(train)
    OneMa[OneMa>0] = 1
    row,col = train.shape
    W = np.zeros((col,col))
    for i in range(col):
        for j in range(col):
            if i==j:
                continue
            wij = np.dot(1/(1+alpha*abs(train[:,i]-train[:,j])),np.multiply(OneMa[:,i],OneMa[:,j]))
            W[i,j] = wij/(np.linalg.norm(OneMa[:,i])*np.linalg.norm(OneMa[:,j])+1.0e-6)
    return W
TICF_WMa = TICF_W(U_I_TMatrix,0.8)

'''5、TICF算法，考虑用户对物品的时间戳，不过是字典数据集，用以和矩阵算法做对比'''
def TICF_WDict(train,alpha):
    ItemNum = dict()
    SimNum = dict()
    for u,i_t in U_IDict.items():
        for i, itimestamp in i_t.items():
            '''计算喜欢该物品的用户数'''
            if i not in ItemNum.keys():
                ItemNum[i] = 0
            ItemNum[i] += 1
            '''计算两两物品之间的同一个用户数'''
            if i not in SimNum.keys():
                SimNum[i] = dict()
            for j, jtimestamp in i_t.items():
                if i==j:
                    continue
                if j not in SimNum[i].keys():
                    SimNum[i][j] = 0
                SimNum[i][j] += 1/(1+alpha*abs(itimestamp-jtimestamp))
    W = dict()
    for i, j_wij in SimNum.items():
        W[i] = dict()
        for j, wij in j_wij.items():
            W[i][j] = wij/np.sqrt(ItemNum[i]*ItemNum[j])
    return W
TICF_WDict = TICF_WDict(U_IDict,0.8)

'''6、TUCF算法，考虑用户对物品的时间戳'''
def TUCF_WMa(train,alpha):
    OneMa = copy.deepcopy(train)
    OneMa[OneMa>0] = 1
    row,col = train.shape
    W = np.zeros((row,row))
    for ui in range(row):
        for uj in range(row):
            if ui==uj:
                continue
            wij = np.dot(1/(1+alpha*abs(train[ui,:]-train[uj,:])),np.multiply(OneMa[ui,:],OneMa[uj,:]))
            W[ui,uj] = wij/(np.linalg.norm(OneMa[ui,:])*np.linalg.norm(OneMa[uj,:])+1.0e-6)
    return W
TUCF_WMa = TUCF_WMa(U_I_TMatrix,0.8)    

'''7、推荐评价'''
'''有以下算法对应的权重，需要分别推荐与评价
Item_CF:ICF_WMa,W
User_CF:UCF_WMa
TItem_CF:TICF_WMa,TICF_WDict
TUser_CF:TUCF_WMa
RecentPOP:PopMa,PopDict
PersonalRANK:iRank,GRank
'''
class CommandCheck(object):
    '''7-1初始化类'''
    def __init__(self):
        self.userNum = 0
        self.testNum = 0
        self.TopN = 0
        self.commandNum = 0
        self.K = 0             #选择相邻的10个用户或者物品
        self.hit = 0            #命中数据
        self.precision = 0     #准确率
        self.recall = 0           #召回率
        
    '''7-2Item_CF算法推荐和评价'''
    def CommandICF(self,W,User,K,TopN):
        global UserList,ItemList,TimeList,U_IDict
        havingItems = U_IDict[User].keys()
        rank = dict()
        for i in havingItems:
            iIdx = ItemList.index(i)
            SimItemIdx = np.argsort(-W[iIdx,:])[0:K]
            for cmdiIdx in SimItemIdx:
                if ItemList[cmdiIdx] in havingItems:
                    continue
                if ItemList[cmdiIdx] not in rank.keys():
                    rank[ItemList[cmdiIdx]] = 0
                rank[ItemList[cmdiIdx]] += W[iIdx,cmdiIdx]
        cmdRANK = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:TopN]
        return cmdRANK
    '''第二种方法：用矩阵的思维来做'''
    def CommandICF2(self,W,User,K,TopN):
        global UserList,ItemList,U_IMatrix
        rank = dict()
        UserIdx = UserList.index(User)
        havingItemsIdx = U_IMatrix[UserIdx,:].nonzero()[0]
        havingItems = [ItemList[x] for x in havingItemsIdx]
        for i in havingItemsIdx:
            W[i,:][W[i,:]<np.sort(W[i,:])[-K]] = 0
        rankW = W[havingItemsIdx,:].sum(axis=0)
        for j in np.argsort(-rankW):
            if ItemList[j] not in havingItems:
                rank[ItemList[j]] = rankW[j]
                #print(ItemList[j],rankW[j])
            if len(rank)==TopN:
                return rank
                break
    def ICFcheck(self,W,testDict,K,TopN):            
        hit = 0
        self.K = K
        self.TopN = TopN
        self.userNum = len(testDict)
        self.commandNum = self.TopN*self.userNum
        for u,i in testDict.items():
            self.testNum += len(i)
            CmdRank = self.CommandICF2(W,u,K,TopN)
            for cmdi ,wi in CmdRank.items():
                if cmdi in i:
                    hit += 1
                    print(u,':',hit)
        self.hit = hit
        self.precision = hit/self.commandNum
        self.recall = hit/self.testNum
        print("precison:{}".format(self.precision))
        print("recall:{}".format(self.recall))

    '''7-3User_CF算法推荐与评价'''
    def CommandUCF(self,W,User,K,TopN):
        global UserList,ItemList,TimeList,U_IDict,U_IMatrix
        rank = dict()
        havingItems = U_IDict[User].keys()
        SimUserIdx = np.argsort(-W[UserList.index(User),:])[0:K]
        UserSim = W[UserList.index(User),SimUserIdx]
        SimUIMatrix = U_IMatrix[SimUserIdx,:]
        SimItem = np.dot(UserSim,SimUIMatrix)
        SimItemIdx = np.argsort(-SimItem)
        for iIdx in SimItemIdx:
            cmdItem = ItemList[iIdx]
            if cmdItem in havingItems:
                continue
            #print(cmdItem)
            rank[cmdItem] = SimItem[iIdx]
            if len(rank) == TopN:
                return rank
                break
    def UCFcheck(self,W,testDict,K,TopN):            
        hit = 0
        self.K = K
        self.TopN = TopN
        self.userNum = len(testDict)
        self.commandNum = self.TopN*self.userNum
        for u,i in testDict.items():
            self.testNum += len(i)
            CmdRank = self.CommandUCF(W,u,K,TopN)
            for cmdi ,wi in CmdRank.items():
                if cmdi in i:
                    hit += 1
                    print(u,':',hit)
        self.hit = hit
        self.precision = hit/self.commandNum
        self.recall = hit/self.testNum
        print("precison:{}".format(self.precision))
        print("recall:{}".format(self.recall))
    
    '''7-4POP流行度算法推荐与评价'''
    def CommandPOP(self,W,User,TopN):
        global UserList,ItemList,U_IMatrix
        rank = dict()
        UserIdx = UserList.index(User)
        havingItemsIdx = U_IMatrix[UserIdx,:].nonzero()[0]
        for index in np.argsort(-W):
            if index in havingItemsIdx:
                continue
            rank[ItemList[index]] = W[index]
            if len(rank)==TopN:
                return rank
                break
    def POPcheck(self,W,testDict,TopN):
        hit = 0
        self.TopN = TopN
        self.userNum = len(testDict)
        self.commandNum = self.TopN*self.userNum
        for u,i in testDict.items():
            self.testNum += len(i)
            CmdRank = self.CommandPOP(W,u,TopN)
            for cmdi ,wi in CmdRank.items():
                if cmdi in i:
                    hit += 1
                    print(u,':',hit)
        self.hit = hit
        self.precision = hit/self.commandNum
        self.recall = hit/self.testNum
        print("precison:{}".format(self.precision))
        print("recall:{}".format(self.recall))

    '''7-5TItem_CF算法推荐和评价'''
    def CommandTICF(self,W,User,K,TopN,T0,beta):
        global UserList,ItemList,U_I_TMatrix
        rank = dict()
        userIdx = UserList.index(User)
        havingiIdx = U_I_TMatrix[userIdx,:].nonzero()[0]
        havingItems = [ItemList[x] for x in havingiIdx]
        Tw = 1/(1+beta*abs(T0 - U_I_TMatrix[userIdx,havingiIdx]))
        for iIdx in havingiIdx:
            W[iIdx,:][W[iIdx,:]<np.sort(W[iIdx,:])[-K]] = 0
        Iw = W[havingiIdx,:]
        rankW = np.dot(Tw,Iw)
        for i in np.argsort(-rankW):
            cmdItem = ItemList[i]
            if cmdItem not in havingItems:
                rank[cmdItem] = rankW[i]
            if len(rank) == TopN:
                return rank
                break
    def TICFcheck(self,W,testDict,K,TopN,T0,beta):            
        hit = 0
        self.K = K
        self.TopN = TopN
        self.userNum = len(testDict)
        self.commandNum = self.TopN*self.userNum
        for u,i in testDict.items():
            self.testNum += len(i)
            CmdRank = self.CommandTICF(W,u,K,TopN,T0,beta)
            for cmdi ,wi in CmdRank.items():
                if cmdi in i:
                    hit += 1
                    print(u,':',hit)
        self.hit = hit
        self.precision = hit/self.commandNum
        self.recall = hit/self.testNum
        print("precison:{}".format(self.precision))
        print("recall:{}".format(self.recall))

    '''7-6TUser_CF算法推荐和评价'''
    def CommandTUCF(self,W,User,K,TopN,T0,beta):
        global UserList, ItemList, U_I_TMatrix
        rank = dict()
        userIdx = UserList.index(User)
        havingiIdx = U_I_TMatrix[userIdx,:].nonzero()[0]
        havingItems = [ItemList[x] for x in havingiIdx]
        SimuIdx = np.argsort(-W[userIdx,:])[0:K]
        Uw = W[userIdx,SimuIdx]
        Iw = 1/(1+beta*abs(T0-U_I_TMatrix[SimuIdx,:]))
        Iw[U_I_TMatrix[SimuIdx,:]==0] = 0
        rankW = np.dot(Uw,Iw)
        for i in np.argsort(-rankW):
            cmdItem = ItemList[i]
            if cmdItem not in havingItems:
                rank[cmdItem] = rankW[i]
            if len(rank) == TopN:
                return rank
                break
    def TUCFcheck(self,W,testDict,K,TopN,T0,beta):            
        hit = 0
        self.K = K
        self.TopN = TopN
        self.userNum = len(testDict)
        self.commandNum = self.TopN*self.userNum
        for u,i in testDict.items():
            self.testNum += len(i)
            CmdRank = self.CommandTUCF(W,u,K,TopN,T0,beta)
            for cmdi ,wi in CmdRank.items():
                if cmdi in i:
                    hit += 1
                    print(u,':',hit)
        self.hit = hit
        self.precision = hit/self.commandNum
        self.recall = hit/self.testNum
        print("precison:{}".format(self.precision))
        print("recall:{}".format(self.recall))

        
'''8、正式程序'''
'''8-1准备测试集字典'''
testDict = dict()
for u,i,s,t in np.array(testSet):
    if u not in testDict.keys():
        testDict[u] = set()
    if i not in testDict[u]:
        testDict[u].add(i)
'''8-3调用类'''
cmd = CommandCheck()
'''8-3-1ICF结果'''
cmd.ICFcheck(ICF_WMa,testDict,10,10)
'''
precison:0.007880794701986755
recall:0.07880794701986756
'''

'''8-3-2UCF结果'''
cmd.UCFcheck(UCF_WMa,testDict,10,10)
'''
precison:0.007963576158940397
recall:0.07963576158940397
'''

'''8-3-3随机游走图推荐'''

'''8-3-4最热门推荐'''
cmd.POPcheck(PopMa,testDict,10)
'''
precison:0.001490066225165563
recall:0.014900662251655629
'''

'''8-3-5TICF结果'''
T0 = np.max(U_I_TMatrix)
cmd.TICFcheck(TICF_WMa,testDict,10,10,T0,0.8)
'''
precison:0.006539735099337749
recall:0.06539735099337748
'''

'''8-3-6TUCF结果'''
cmd.TUCFcheck(TUCF_WMa,testDict,10,10,T0,0.8)
'''
precison:0.002980132450331126
recall:0.029801324503311258
'''


######################################
#                                    #
#       基于社交网络的推荐系统        #
#                                    #
######################################
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import operator
os.chdir(r"D:\mywork\test\command\social_community")
'''用户-物品数据'''
with open("user_artists.dat","r",errors="ignore") as f:
    content1 = f.readlines()
U_Idata = np.array([x.split() for x in content1][1:])
'''用户-朋友数据'''
with open(r"user_friends.dat","r",errors="ignore") as f:
    content2 = f.readlines()
U_Fdata = np.array([x.split() for x in content2][1:])

UserList = list(set(U_F[:,0]))
ItemList = list(set(U_I[:,1]))
'''计算用户-物品矩阵'''
U_IMatrix = np.zeros((len(UserList),len(ItemList)))
for u, i, w in U_Idata:
    uIdx = UserList.index(u)
    iIdx = ItemList.index(i)
    U_IMatrix[uIdx,iIdx] = 1
'''计算用户-朋友矩阵'''
U_FMatrix = np.zeros((len(UserList),len(UserList)))
for u,f in U_Fdata:
    uIdx = UserList.index(u)
    fIdx = UserList.index(f)
    U_FMatrix[uIdx,fIdx] = 1

def cosDist(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

'''1、计算用户熟悉度和相似度'''
def CalSim(train):
    global UserList, ItemList
    row,col = train.shape
    Sim = np.zeros((row,row))
    for i in range(row):
        for j in range(row):
            if i==j:
                continue
            Sim[i,j] = cosDist(train[i,:],train[j,:])
    return Sim
SimUF = CalSim(U_FMatrix)
SimUI = CalSim(U_IMatrix)
'''2、根据熟悉的用户将喜欢的物品推荐给目标用户'''
def Command(user,SimUF,SimUI,U_IMatrix,K):
    global UserList,ItemList
    rank = dict()
    '''2-1用户已经喜欢的物品'''
    uIdx = UserList.index(user)
    havingiIdx = U_IMatrix[uIdx,:].nonzero()[0]
    havingItems = [ItemList[x] for x in havingiIdx]
    '''2-1根据用户之间的熟悉度进行推荐'''
    SimFriendsIdx = np.argsort(-SimUF[uIdx,:])[:K]
    SimFriendsMa = SimUF[uIdx,SimFriendsIdx]
    FriendsItemsMa = U_IMatrix[SimFriendsIdx,:]
    rankMa = np.dot(SimFriendsMa,FriendsItemsMa)
    for iIdx in np.argsort(-rankMa):
        if iIdx in havingiIdx:
            continue
        item = ItemList[iIdx]
        rank[item] = rankMa[iIdx]
    '''2-2根据用户相似度进行推荐'''
    SimUserIdx = np.argsort(-SimUI[uIdx,:])[:K]
    SimUserMa = SimUF[uIdx,SimUserIdx]
    UserItemsMa = U_IMatrix[SimUserIdx,:]
    rankMa2 = np.dot(SimUserMa,UserItemsMa)
    for iIdx in np.argsort(-rankMa2):
        if iIdx in havingiIdx:
            continue
        item2 = ItemList[iIdx]
        if item2 not in rank.keys():
            rank[item2] = 0
        rank[item2] += rankMa2[iIdx]
    return rank
cmdRank = Command('22',SimUF,SimUI,U_IMatrix,10)


######################################
#                                    #
#        推荐好友离线实验测试         #
#                                    #
######################################
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import operator
os.chdir(r"D:\mywork\test\command\social_Slashdot")

'''1、读取数据并划分训练集和测试集'''
with open("Slashdot0902.txt","r") as f:
    content = f.readlines()
data = np.array([[int(y) for y in x.split()] for x in content[4:]])
np.random.seed(1234)
train = dict()      #用户关注其他用户的集合
test = dict()       #测试集
focus = dict()      #用户被其他用户关注的集合

M = 10
M0 = 0
for User,focusU in data:
    if np.random.randint(0,M) == M0:
        if User not in test.keys():
            test[User] = set()
        test[User].add(focusU)
    else:
        if User not in train.keys():
            train[User] = set()
        train[User].add(focusU)
        if focusU not in focus.keys():    
            focus[focusU] = set()
        focus[focusU].add(User)
'''2、分别计算相似度，内存不足，无法用array'''
'''2-1用户a和用户b关注客户集合的相似度'''
SimOut = dict()
for userA,AfocusU in train.items():
    for userB,BfocusU in train.items():
        if userA==userB:
            continue
        if userA not in SimOut.keys():
            SimOut[userA] = dict()
        if userB not in SimOut[userA].keys():
            SimOut[userA][userB] = len(AfocusU&BfocusU)/np.sqrt(len(AfocusU)*len(BfocusU))
    print(userA)
    if len(SimOut)==3000:
        break

'''2-2用户a和用户b被关注客户集合的相似度'''
SimIn = dict()
for userA, AbefocusU in focus.items():
    for userB, BbefocusU in focus.items():
        if userA==userB:
            continue
        if userA not in SimIn.keys():
            SimIn[userA] = dict()
        if userB not in SimIn[userA].keys():
            SimIn[userA][userB] = len(AbefocusU&BbefocusU)/np.sqrt(len(AbefocusU)*len(BbefocusU))
    print(userA)

'''2-3用户a关注的客户集合和用户b被关注客户集合的相似度'''
SimOutIn = dict()
for userA,AfocusU in train.items():
    for userB, BbefocusU in focus.items():
        if userA==userB:
            continue
        if userA not in SimOutIn.keys():
            SimOutIn[userA] = dict()
        if userB not in SimOutIn[userA].keys():
            SimOutIn[userA][userB] = len(AfocusU&BbefocusU)/np.sqrt(len(AfocusU)*len(BbefocusU))
    print(userA)

'''3、推荐好友'''
def CommandFriend(u,w,K):
    rank = dict()
    friends = train[u]
    Sim = sorted(w[u].items(), key=operator.itemgetter(1), reverse=True)
    for f,wf in Sim:
        if f in friends:
            continue
        rank[f] = wf
        if len(rank) == K:
            return rank
            break
        
'''4、评价好友推荐效果'''
def evaluate(test,w,K):
    allcommand = 0
    alltest = 0
    hit = 0
    for u, testFriends in test.items():
        try:
            rank = CommandFriend(u,w,K)
            alltest += len(testFriends)
            allcommand += len(rank)
            for cmdf in rank.keys():
                if cmdf in testFriends:
                    hit += 1
            print('{}已推荐，hit{}'.format(u,hit))            
        except:
            print('{}不在其中'.format(u))
    precision = hit/allcommand
    recall = hit/alltest
    print("precision:%.5f;"%precision,"recall:%.5f"%recall)
    return precision,recall  
'''4-1对Out评价，内存原因，关系矩阵只取了前3000个'''
evaluate(test2,SimOut,10)





