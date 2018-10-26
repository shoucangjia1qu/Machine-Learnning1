# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:10:12 2018

@author: ecupl
"""

import os
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from selenium import webdriver
import json
from urllib.request import urlopen, quote
import requests
import ast
from bs4 import BeautifulSoup
import copy

os.chdir(r'D:\mywork\test')


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
        


######################正式程序##############################
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
dc.train(point,500)
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
import pickle
curpath=os.getcwd()
with open(curpath+"\\centerPoint.dat","wb") as obj:
    pickle.dump(cpData,obj)


'''6、读取对象'''
with open(curpath+"\\centerPoint.dat","rb") as f:
    cps=pickle.load(f)







