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

os.chdir(r'D:\mywork\test')


#构造算法，最密集点聚类
class disClustor(object):
    '''1、初始化'''
    def __init__(self):         #构造方法
        self.centerPoint={}     #最终分类的中心点
        self.rangeDist=0        #指定的分类半径
        
    '''2、计算两点之间的距离'''
    def haversine(self,lng1,lat1,lng2,lat2):
        rlng1,rlat1,rlng2,rlat2 = map(radians,[lng1,lat1,lng2,lat2])
        dlng = rlng2-rlng1
        dlat = rlat2-rlat1
        a = sin(dlat/2)**2 + cos(rlat1)*cos(rlat2)*sin(dlng/2)**2
        c = 2*asin(sqrt(a))
        r = 6371
        return(c*r*1000)
    
    '''3、生成所有点与点之间的距离'''
    def dictdistance(self,point,length):      
        pointNum={}     #圈内人数
        pointAvgDist={} #生成每个圈内平均距离
        everyDist={}    #生成每个圈
        for i in point:
            subdist={}      #生成空的子字典
            alldists=0      #圈内到中心点的距离
            cstnum=0        #圈内客户数
            for j in point:
                dist = self.haversine(i[0],i[1],j[0],j[1])     #调用距离函数
                if dist<=length:
                    subdist[j]=dist
                    alldists+=dist
                    cstnum+=1
            if cstnum > 1:        #选取不止一个点
                pointAvgDist[i]=alldists/(cstnum-1)
                pointNum[i]=cstnum
            everyDist[i]=subdist
        return pointNum,pointAvgDist,everyDist
    
    '''4、删除已分圈的点'''
    def delpoints(self,point,delpointlist):
        removelist=[point.index(i) for i in point if i in delpointlist]     #需要删除的元素index
        x=0
        for y in removelist:
            point.pop(y-x)
            x+=1
        return point
    
    '''5、创建中心点'''
    def buildpoint(self,point,length):
        #获取圈半径
        self.rangeDist = length
        #抛出所有点信息
        pointNum,pointAvgDist,everyDist=self.dictdistance(point,self.rangeDist)
        #判断是否需要迭代
        lp = list(pointNum.values())
        if len(lp) == lp.count(min(lp)):
            return "完成"
        else:
            #构建圈中点信息
            subpoint={}         #圈的子字典
            maxNum = max(pointNum.values())     #最大人数
            subpoint['roundNum']=maxNum
            maxPoint = list(pointNum.keys())[list(pointNum.values()).index(maxNum)]     #最大人数的中心点辐射的圈
            pointDistance = pointAvgDist.get(maxPoint)      #最大人数圈的平均距离
            subpoint['roundDist']=pointDistance
            roundpts = list(everyDist.get(maxPoint).keys())
            subpoint['roundPoints']=roundpts
            self.centerPoint[maxPoint] = subpoint
            #删除圈中点
            point = self.delpoints(point,roundpts)
            #继续迭代
            self.buildpoint(point,length)

'''爬取位置信息'''
def getlatlng1(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = 'v9OaBDfj21B9RYMrH93qVO998RZiHXU9' # 百度地图ak，具体申请自行百度，提醒需要在“控制台”-“设置”-“启动服务”-“正逆地理编码”，启动
    address = quote(address) # 由于本文地址变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + address  + '&output=' + output + '&ak=' + ak 
    
    #第一种方法，用requests搭配bs4，再用ast转换成字典
    html=requests.get(uri).text.encode("utf-8-sig")
    sp=BeautifulSoup(html,"html.parser")
    jsondata=ast.literal_eval(sp.text)  
    lng=jsondata['result']['location']['lng']
    lat=jsondata['result']['location']['lat']
    return(lng,lat)




'''爬取经纬度'''
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
'''去重和保存'''
location=location.drop_duplicates(['lng','lat'])
location.reset_index(drop=True,inplace=True)     
location.to_csv("ccbcstsite.csv")
'''读取位置信息'''
location = pd.read_csv("ccbcstsite.csv",encoding="GBK")
point=[ i for i in zip(location['lng'],location['lat'])]


#实例化
dc=disClustor()
dc.buildpoint(point,500)
cps=dc.centerPoint
cpvaluelist = list(cps.values())
cpkeylist = list(cps.keys())
'''对象持久化'''
import pickle
curpath=os.getcwd()
with open(curpath+"\\centerPoint.dat","wb") as obj:
    pickle.dump(cps,obj)
'''中心点导出到excel，将地址信息补全''' 
num=[i.get('roundNum') for i in cpvaluelist]
avgdist=[i.get('roundDist') for i in cpvaluelist]
lng=[i[0] for i in cpkeylist]
lat=[i[1] for i in cpkeylist]
df = pd.DataFrame({'lng':lng,'lat':lat,'num':num,'avgdist':avgdist})
df2=pd.merge(df, location, on=('lng','lat'), how='left')   
df2.drop_duplicates(subset=['lng','lat'],inplace=True)
df2.reset_index(drop=True,inplace=True)
df2.to_csv("centerPoints.csv")
'''圈内点'''
roundPoints=[i.get('roundPoints') for i in cpvaluelist]
allpts=pd.DataFrame()
for i in range(len(avgdist)):
    for j in roundPoints[i]:
        address=df2.adr[i]
        allpts=allpts.append([[address,j[0],j[1]]])
allpts=allpts.rename(columns={0:'center',1:'lng',2:'lat'})
location.drop_duplicates(subset=['lng','lat','adr'],inplace=True)
allpts=pd.merge(allpts, location, on=('lng','lat'), how='left')   
allpts.to_csv("allpts.csv")

'''读取对象'''
with open(curpath+"\\centerPoint.dat","rb") as f:
    cps=pickle.load(f)

