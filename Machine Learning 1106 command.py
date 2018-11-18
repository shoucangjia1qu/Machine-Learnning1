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

os.chdir(r'D:\mywork\test\command\regItemCold')
curpath = os.getcwd()
#with open("users.txt","rb") as f:
#    users = f.readlines
'''读入数据'''
users = pd.read_csv('usersha1-profile.tsv',sep='\t',header=None)
users = users.rename(columns={0:'id',1:'gender',2:'age',3:'country',4:'data'})
music = pd.read_csv('usersha1-artmbid-artname-plays.tsv',sep='\t',header=None)
music = music.rename(columns={0:'id',1:'artistId',2:'artistName',3:'times'})

######################################
#                                    #
#            冷启动问题               #
#                                    #
######################################
'''一、利用用户注册信息'''
'''1、区分出不同类别的分类变量'''
'''1-1gender'''
users.gender=users.gender.fillna('o')
gender = list(set(users.gender))
'''1-2country'''
countryCount = users.country.value_counts()
countryList = countryCount[countryCount<=5000].index.tolist()
users.country[users['country'].isin(countryList)] = 'other'
country = list(users.country.value_counts().index)
'''1-3age'''
bins = [0,10,20,30,40,50,60,70,80,90,100]
users.age = pd.cut(users.age,bins,labels=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10'])
users.age=users.age.cat.add_categories('a0')
users.age = users.age.fillna('a0')
age = list(users.age.value_counts().index)

'''2、按类别划分不同客户群体'''
userCate=[]
userDict = dict()
n = 0
for i in gender:
    for j in age:
        for k in country:
            userCate.append(i+j+k)
            userDict[n]=users.id[users['gender']==i][users['age']==j][users['country']==k].tolist()
            n += 1
'''3、选取用户的训练集和测试集'''
train = []
test = []
np.random.seed(1234)
for i in users.id:
    if np.random.randint(0,10) == 0:
        test.append(i)
    else:
        train.append(i)
'''4、训练每个用户所属客户群体的艺术家偏好'''
music.artistName = music.artistName.fillna('nothing')
musicList = list(music.artistName.value_counts().index)
userCate_Music = np.zeros((len(musicList),len(userCate)))
music_array = np.array(music)
for x in range(len(music_array)):
    U = music_array[x,0]
    '''4-1确定用户所属群体的Index'''
    if U in train:
        temp = users[users['id']==U]
        Ugender = temp.gender.values[0]
        Uage = temp.age.values[0]
        Ucountry = temp.country.values[0]
        userCateIdx = userCate.index(Ugender+Uage+Ucountry)
        '''4-1确定艺术家的Index'''
        artistIdx = musicList.index(music_array[x,2])
        '''4-2相应位置+1'''
        userCate_Music[artistIdx,userCateIdx] += 1

'''5、根据测试用户所属群体推荐'''
def command(u,users,userCate,musicList,userCate_Music,TopN):
    '''5-1确定测试用户所属群体的Index'''
    testtemp = users[users['id'] == u]
    Tgender = testtemp.gender.values[0]
    Tage = testtemp.age.values[0]
    Tcountry = testtemp.country.values[0]
    TuserCateIdx = userCate.index(Tgender+Tage+Tcountry)
    '''5-2确定用户所属群体的音乐推荐'''
    MuList = userCate_Music[:,TuserCateIdx]
    MuIdx = np.argsort(-MuList)[:,TopN]
    '''5-3返回推荐的音乐'''
    rank = [musicList[x] for x in MuIdx]
    return rank

'''6、根据细颗粒度（性别、年龄、国籍）划分的客群评估成效'''   
hit = 0
allcommand = 0
alltest = 0
allMusic = 0
cmdMusic = set()
for uid in test:
    '''6-1推荐的TopN产品'''
    rank = command(uid,users,userCate,musicList,userCate_Music,10)
    '''6-2计算全部推荐产品的数量'''
    allcommand += len(rank)
    '''6-3计算用户实际的产品数量'''
    alltest += len(testMusic)
    '''6-4计算全部产品的数量'''
    allMusic = len(musicList)
    '''6-5计算推荐产品的命中数量和产品集合'''
    for j in rank:
        if j in testMusic:
            hit += 1
        cmdMusic.add(i)
    '''6-6输出离线指标'''
    precision = hit/allcommand
    recall = hit/alltest
    cover = len(cmdMusic)/allMusic


######################################
#                                    #
#           利用用户标签              #
#                                    #
######################################
import numpy as np
import pandas as pd
import os, operator, pickle
import matplotlib.pyplot as plt

os.chdir(r"D:\mywork\test\command\Tag")
with open("user_taggedbookmarks-timestamps.dat","r",encoding="utf-8",errors="ignore" ) as f:
    content = f.readlines()
data = np.array([x.split() for x in content][1:])
single_UI = pd.DataFrame(data[:,0:2]).drop_duplicates()
single_UI = np.array([list(single_UI.iloc[:,0]),list(single_UI.iloc[:,1])]).T
'''按照用户-物品划分训练集和测试集'''
UItrain = []
UItest = []
np.random.seed(1234)
for ui in single_UI:
    if np.random.randint(0,10) == 0:
        UItest.append(ui.tolist())
    else:
        UItrain.append(ui.tolist())
'''再将标签加入，生成用户-物品-标签的训练集和测试集'''
trainIdx = []
testIdx = []
for i in range(len(data)):
    if data[i,0:2].tolist() in UItest:
        testIdx.append(i)
    else:
        trainIdx.append(i)
train = data[trainIdx]
test = data[testIdx]
'''得到两两矩阵'''
userList = list(set(train[:,0]))
bookList = list(set(train[:,1]))
tagList = list(set(train[:,2]))
'''U_T,U_I,T_I矩阵'''
U_T = np.zeros((len(userList),len(tagList)))
U_I = np.zeros((len(userList),len(bookList)))
T_I = np.zeros((len(tagList),len(bookList)))
for u,i,t in train[:,0:3]:
    uIdx = userList.index(u)
    iIdx = bookList.index(i)
    tIdx = tagList.index(t)
    U_T[uIdx,tIdx] += 1
    U_I[uIdx,iIdx] += 1
    T_I[tIdx,iIdx] += 1
'''得出预测的矩阵'''
Wui = np.dot(U_T,T_I)
#将已有的UI项目变为0
Wui[U_I>0] = 0
'''command推荐'''
def command(u,topN):
    global bookList,userList,Wui
    rank = {}
    uIdx = userList.index(u)
    itemsIdx = np.argsort(-Wui[uIdx])[0:topN]
    for itemIdx in itemsIdx:
        item = bookList[itemIdx]
        rank[item] = Wui[uIdx,itemIdx]
    return rank
'''夹角余弦'''
def cosDist(v1,v2):
    sim = np.dot(v1,v2)/np.linalg.norm(v1)*np.linalg.norm(v2)
    return sim
'''推荐评价'''
topN = 100
allpopular = dict()
allItems = len(bookList)
UI_test = np.array(UItest)
testUserList = list(set(UI_test[:,0]))
alltest = len(UI_test)
allcommand = len(testUserList)*topN
hit = 0
cmdpopular = 0
cmdItems = set()
cmdSim = 0
for i in range(len(bookList)):
    if bookList[i] not in allpopular.keys():
        allpopular[bookList[i]] = 0
    allpopular[bookList[i]] += len(U_I[:,i][U_I[:,i]>0])
for user in testUserList:
    try:
        rank = command(user,topN)
        testItems = UI_test[UI_test[:,0]==user][:,1].tolist()
        for cmditem in rank.keys():
            if cmditem in testItems:
                hit += 1
            cmdpopular += allpopular[cmditem]
            cmdItems.add(cmditem)
    except:
        print("客户不在训练集中")
I_T = T_I.T
for i in cmdItems:
    for j in cmdItems:
        if i==j:
            continue
        iIdx = bookList.index(i)
        jIdx = bookList.index(j)
        cmdSim += cosDist(I_T[iIdx],I_T[jIdx])


precision = hit/allcommand
recall = hit/alltest
pop = cmdpopular/allcommand
cover = len(cmdItems)/allItems
multi = cmdSim/len(cmdItems)

print('''precision:{}\t
         recall:{}\t
         pop:{}\t
         cover:{}\t
         multi:{}'''.format(precision,recall,pop,cover,multi))

'''算法改进1：引入惩罚因子'''
#新的矩阵(U_T和T_I)
U_TNew = np.zeros((len(userList),len(tagList)))
T_INew = np.zeros((len(tagList),len(bookList)))
#惩罚因子，某标签被多少用户使用过
for tIdx in range(len(tagList)):
    Ntu = len(U_T[U_T[:,tIdx]>0])
    U_TNew[:,tIdx] = U_T[:,tIdx]/np.log(1+Ntu)
#惩罚因子，某物品被多少用户打过标签
for iIdx in range(len(bookList)):
    Niu = len(U_I[U_I[:,iIdx]>0])
    T_INew[:,iIdx] = T_I[:,iIdx]/np.log(1+Niu)
'''得出预测的矩阵'''
Wui2 = np.dot(U_TNew,T_INew)
#将已有的UI项目变为0
Wui2[U_INew>0] = 0

'''算法改进2：改进数据稀疏性'''
#先计算标签相似度
TagSim = np.zeros((len(tagList),len(tagList)))
for i in range(len(tagList)):
    for j in range(len(tagList)):
        if i==j:
            continue
        TagSim[i,j] = cosDist(T_I[i],T_I[j])

'''算法改进3：标签清理'''

######################################
#                                    #
#     用户标签中基于图的推荐算法        #
#                                    #
######################################
import numpy as np
import pandas as pd
import os, operator, pickle
import matplotlib.pyplot as plt

#Simple Tag Graph
#创建Graph{U{I:n}}，凡是用户为物品打过标签的都+1。
graphData = data[0:5000,0:3]        #只取了5000条数据，U,I,T
def buildGraph(data):
    '''生成图字典'''
    graphDict = dict()
    for u,i,t in data:
        if u not in graphDict.keys():
            graphDict[u] = dict()
        if i not in graphDict[u].keys():
            graphDict[u][i] = 0
        graphDict[u][i] += 1
        if i not in graphDict.keys():
            graphDict[i] = dict()
        if u not in graphDict[i].keys():
            graphDict[i][u] = 0
        graphDict[i][u] += 1
    '''生成图矩阵'''
    uList = list(set(data[:,0]))
    iList = list(set(data[:,1]))
    tList = list(set(data[:,2]))
    graphMa = np.zeros((len(uList),len(iList)))
    for u,i,t in data:
        uIdx = uList.index(u)
        iIdx = iList.index(i)
        tIdx = tList.index(t)
        graphMa[uIdx,iIdx]  += 1
    return graphDict,graphMa

graphDict, graphMa = buildGraph(graphData)

'''基于词典的迭代方法'''
def PersonalRank(G,alpha,root,Maxiter):
    '''1、初始化rank,初始节点为1，其他为0'''
    rank = dict()
    rank = {x:0 for x in G.keys()}
    rank[root] = 1
    '''2、开始迭代，有alpha的概率继续往下走，回到原节点的概率为(1-alpha)'''
    for i in range(Maxiter):
        '''2-1初始化'''
        temp = {x:0 for x in G.keys()}
        '''2-2遍历每个主节点和到子节点的所有路径'''
        for leaf, subpath in G.items():
            for subleaf, path in subpath.items():
                '''2-3计算主节点到子节点的概率'''
                temp[subleaf] += alpha*rank[leaf]/(1.0*len(subpath))
        '''2-4加上回到原节点的概率'''
        temp[root] += 1-alpha
        '''2-5替换为最新的rank'''
        rank = temp
    '''3、返回结果并打印'''
    return rank

rank = PersonalRank(graphDict,0.85,'8',100)

'''基于矩阵的迭代方法'''
def PersonalRank2(GMa,alpha,rootIdx,Maxiter):
    row,col = GMa.shape
    uRank = np.zeros((row,1))
    iRank = np.zeros((1,col))
    uRank[rootIdx,0] = 1
    proI = np.zeros((row,col))
    proU = np.zeros((row,col))
    for c in range(col):
        proI[:,c] = GMa[:,c]/GMa.sum(axis=1)
    for r in range(row):
        proU[r,:] = GMa[r,:]/GMa.sum(axis=0)
    for i in range(Maxiter):
        itemp = np.dot(alpha*uRank.T,proI)
        utemp = np.dot(alpha*iRank,proU.T)
        utemp[0,rootIdx] += 1-alpha
        iRank = itemp
        uRank = utemp.T
    return iRank,uRank

iRank,uRank = PersonalRank2(graphMa,0.85,8,100)













