# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:10:12 2018

@author: ecupl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from selenium import webdriver
import json
from urllib.request import urlopen, quote
import requests
import ast
from bs4 import BeautifulSoup
import copy
import pickle

os.chdir(r'D:\工作资料\分行工作\交接\其他\创新\大数据项目\位置信息挖掘\结项材料')
curpath=os.getcwd()


#构造算法类，最密集点聚类
class disClustor(object):
    '''1、初始化'''
    def __init__(self):
        '''自定义属性'''         
        self.rangeDist=0        #指定的分类半径S
        self.trainSet = 0       #位置数据集
        self.distMatrix = 0     #各点之间的距离矩阵
        '''系统迭代中生成的'''
        self.centerPoint = []   #最终分类的中心点列表
        self.pointNumber = []   #最终各个圈中的点数量列表
        self.circlePoints = []  #最终分类圈中的点列表
        self.circleDists = []   #最终分类圈中的各点到中心点距离的列表
        
    '''2、计算两点之间的距离'''
    def haversine(self,lng1,lat1,lng2,lat2):
        rlng1,rlat1,rlng2,rlat2 = map(radians,[lng1,lat1,lng2,lat2])
        dlng = rlng2-rlng1
        dlat = rlat2-rlat1
        a = sin(dlat/2)**2 + cos(rlat1)*cos(rlat2)*sin(dlng/2)**2
        c = 2*asin(sqrt(a))
        r = 6371
        return(c*r*1000)
    
    '''3、生成点之间的距离矩阵'''
    def distMa(self,train):
        m = len(train)
        dataMa = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                dataMa[i,j] = self.haversine(train[i][0],train[i][1],train[j][0],train[j][1])
        return dataMa
    
    '''4、找到圈内点最多的圈、点、距离'''
    def findCircle(self,dataMatrix,r):
        bestCircle = 0          #最多圈内个数
        bestDistList = []       #最多圈的距离列表
        bestIndexList = []      #最多圈的位置列表
        bestRow = 0             #最多圈的位置
        m,n = dataMatrix.shape
        for i in range(m):
            DistList = []
            IndexList = []
            for j in range(n):
                if dataMatrix[i,j] <= r:
                    DistList.append(dataMatrix[i,j])
                    IndexList.append(j)
                if len(DistList) > bestCircle:
                    bestCircle = len(DistList)
                    bestDistList = DistList
                    bestIndexList = IndexList
                    bestRow = i
        return bestCircle,bestDistList,bestIndexList,bestRow
    
    '''5、主函数'''
    def train(self,train,r):
        self.trainSet = train
        self.rangeDist = r
        '''计算点与点之间的距离'''
        dataDist = self.distMa(train)
        self.distMatrix = dataDist

        '''主循环：找到点最多的圈并开始迭代'''
        while True:
            bestCircle,bestDistList,bestIndexList,bestRow = self.findCircle(dataDist,r)
            self.pointNumber.append(bestCircle)         #圈内数量
            self.centerPoint.append(train[bestRow])      #中心点
            self.circleDists.append(bestDistList)       #圈内各点到中心点的距离
            self.circlePoints.append([train[i] for i in bestIndexList])     #圈内所有的点
            '''结束循环条件：圈内数量小于3个点'''
            if bestCircle < 3:
                break
            '''继续迭代，删除上一轮迭代圈中的点'''
            dataDist = np.delete(dataDist,bestIndexList,axis=0)
            dataDist = np.delete(dataDist,bestIndexList,axis=1)
            train = np.delete(train,bestIndexList,axis=0)
        
#构造Boltzmann机，模拟退火算法规划最优路径
class Boltzmann(object):
    '''构造属性'''
    def __init__(self):
        self.T0 = 1000               #初始温度
        self.r = 0.97                #退火系数
        self.maxIter = 2000          #最大迭代次数
        self.dataSet = 0             #数据集
        self.distMa = 0              #距离矩阵
        self.pathList = []           #每次迭代的路径列表
        self.distList = []           #每次迭代的总距离
        self.bestIter = 0            #最优迭代次数
        self.bestPath = []           #最优路径
        self.bestDist = []           #最优路径的距离
    
    '''两点之间球面距离公式'''
    def cirdist(self,v1,v2):
        rlng1,rlat1,rlng2,rlat2 = map(radians,[v1[0],v1[1],v2[0],v2[1]])
        dlng = rlng2-rlng1
        dlat = rlat2-rlat1
        a = sin(dlat/2)**2 + cos(rlat1)*cos(rlat2)*sin(dlng/2)**2
        c = 2*asin(sqrt(a))
        r = 6371
        return(c*r*1000)
    
    '''玻尔兹曼系数'''
    def boltz(self,dist1,dist0,T):
        return(np.exp(-(dist1-dist0)/T))        #逻辑回归函数x<0
    
    '''计算当前位次的距离和'''
    def allDists(self,distMatrix,pathIndex):
        distance = 0
        n = len(pathIndex)
        for i in range(n-1):
            distance += distMatrix[pathIndex[i],pathIndex[i+1]]
        distance += distMatrix[pathIndex[0],pathIndex[n-1]]
        return distance
    
    '''调换路径下标'''
    def changeIndex(self,pathIndex):
        N = len(pathIndex)
        '''要么两两对调'''
        if np.random.rand() < 0.25:
            points = np.floor(np.random.rand(2)*N)      #下取整
            newpathIndex = copy.deepcopy(pathIndex)
            newpathIndex[int(points[0])] = pathIndex[int(points[1])]
            newpathIndex[int(points[1])] = pathIndex[int(points[0])]
        else:
            '''整段位移互换'''
            points = np.floor(np.random.rand(3)*N)
            points.sort()
            a = int(points[0])
            b = int(points[1])
            c = int(points[2])
            if a!=b and b!=c:
                newpathIndex = copy.deepcopy(pathIndex)
                newpathIndex[a:c+1] = pathIndex[b:c+1] + pathIndex[a:b]
            else:
                newpathIndex = self.changeIndex(pathIndex)
        return newpathIndex
            
    
    '''训练函数'''
    def train(self,data):
        self.train = data
        #计算距离矩阵
        m=data.shape[0]
        distMatrix = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i==j:
                    continue
                distMatrix[i,j] = self.cirdist(data[i,:],data[j,:])
        self.distMa = distMatrix
        #初始化距离
        pathIndex0 = list(range(m))
        np.random.shuffle(pathIndex0)
        dist0 = self.allDists(distMatrix,pathIndex0)
        #开始循环迭代
        T = self.T0
        steps = 0
        while steps < self.maxIter:
            substep = 0
            while substep<m:
                pathIndex1 = self.changeIndex(pathIndex0)
                dist1 = self.allDists(distMatrix,pathIndex1)
                if dist1<dist0:
                    dist0 = dist1
                    pathIndex0 = pathIndex1
                    self.pathList.append(pathIndex0)
                    self.distList.append(dist0)
                    self.bestIter += 1
                else:
                    if np.random.rand()<self.boltz(dist1,dist0,T):
                        dist0 = dist1
                        pathIndex0 = pathIndex1
                        self.pathList.append(pathIndex0)
                        self.distList.append(dist0)
                        self.bestIter += 1
                substep += 1
            steps += 1
            T = T*self.r
        self.bestDist = min(self.distList)
        self.bestPath = self.pathList[np.argmin(self.distList)]
    


######################正式程序##############################
'''第一部分：聚类找到高端圈'''
'''1、爬取位置信息-构造函数'''
def getlatlng1(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = '???' # 百度地图ak，具体申请自行百度，提醒需要在“控制台”-“设置”-“启动服务”-“正逆地理编码”，启动
    address = quote(address) # 由于本文地址变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + address  + '&output=' + output + '&ak=' + ak 
    
    #第一种方法，用requests搭配bs4，再用ast转换成字典
    html=requests.get(uri).text.encode("utf-8-sig")
    sp=BeautifulSoup(html,"html.parser")
    jsondata=ast.literal_eval(sp.text)  
    lng=jsondata['result']['location']['lng']
    lat=jsondata['result']['location']['lat']
    return(lng,lat)


'''2、爬取经纬度'''
n=0
location = pd.DataFrame(columns=['adr','lng','lat'])
for i in address1:
    try:
        lng=getlatlng2(i)[0]
        lat=getlatlng2(i)[1]
        location=location.append({"adr":"{}".format(i),
                                  "lng":"{}".format(lng),"lat":"{}".format(lat)},ignore_index=True)
        print(n)
        n+=1
    except:
        print('%s取不到经纬度。'%i)


'''3-1、去重和保存'''
location=location.drop_duplicates(['lng','lat'])
location.reset_index(drop=True,inplace=True)     
location.to_csv("ccbcstsite.csv")
'''3-2读取位置信息'''
location = pd.read_csv("ccbcstsite.csv",encoding="GBK")
point=np.array(location.iloc[:,1:3])


'''4、函数实例化，并进行训练，得到业务所需数据集'''
dc=disClustor()
dc.train(point[:2000],500)
centerPoints = dc.centerPoint
circleDists = dc.circleDists
pointNumber = dc.pointNumber
circlePoints = dc.circlePoints


'''5、对象持久化'''
cpData = dict()
for i in range(len(centerPoints)):
    cpData[i] = dict()
    cpData[i]['point'] = centerPoints[i].tolist()
    cpData[i]['quantity'] = pointNumber[i]
    cpData[i]['avgdist'] = sum(circleDists[i])/(len(circleDists[i])-1)
#按圈内点数量倒序排列
with open(curpath+"\\centerPoint.dat","wb") as obj:
    pickle.dump(cpData,obj)


'''6、读取对象'''
with open(curpath+"\\centerPoint.dat","rb") as f:
    cps=pickle.load(f)


'''第二部分：找到目标客户'''
'''7、先找到小区位置，再根据通配找到目标客户'''
target = pd.read_csv("lianjia_xiaoqu_20180811.csv",encoding="GBK")
targetData = np.array(target.iloc[:,3:5])
targetMatrix = np.zeros((targetData.shape[0],len(cps)))
m,n = targetMatrix.shape
points = [i.get("point") for i in list(cps.values())]
#生成目标距离矩阵
for i in range(m):
    for j in range(n):
        targetMatrix[i,j] = dc.haversine(points[j][0],points[j][1],targetData[i][0],targetData[i][1])
#选出500米内小区
targetdict = dict()
for col in range(n):
    targetdict[col] = []
    for row in range(m):
        if targetMatrix[row,col]<=500:
            targetdict[col].append(targetData.tolist()[row])
#持久化1
with open(curpath+"\\targetPoints.dat","wb") as obj:
    pickle.dump(targetdict,obj)
#持久化2,只选取目标圈内超过20个私行客户:
targetList = []
for idx,points in targetdict.items():
    if idx<=12:
        targetList.extend(points)
targetDF = pd.DataFrame(targetList,columns=('lng','lat'))
targetDF.to_csv("targetDF.csv")


'''第三部分：对目标客户规划最短路径'''
'''8、对圈内目标客户进行最短路径规划'''
bm = Boltzmann()
dataSet = np.array(targetDF.iloc[:20,:])
bm.train(dataSet)       #需要输入array格式
print(bm.bestIter)      #最短路径迭代次数
print(bm.bestDist)      #最短路径距离
print(bm.bestPath)      #最短路径下标
'''路径距离变化可视化'''
dists = bm.distList
plt.figure()
plt.plot(range(len(dists)),dists)
plt.show()
'''最优路径可视化，经纬度差异不大，故扩大了100倍'''
bestPath = bm.bestPath
x = [dataSet[i,0]*100 for i in bestPath]
y = [dataSet[i,1]*100 for i in bestPath]
plt.figure()
plt.scatter(x,y,c='r',linewidths=5)
plt.plot(x,y,'b--')
i = 0
for xl,yl in zip(x,y):
    plt.annotate("{}".format(bestPath[i]), xy=(xl,yl))
    i += 1
plt.show()





