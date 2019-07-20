# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:52:06 2019

@author: ecupl
"""


#data
with open(r'D:\mywork\test\UCI_data\iris.data') as f:
    data = f.readlines()
x = np.array([[float(i) for i in row.split(",")[:-1]] for row in data[:-1]])
y = np.array([row.split(",")[-1].splitlines() for row in data[:-1]])
yvalue = set(y[:,0])
n = 0
for value in yvalue:
    y[y==value] = n
    n += 1
y = y.astype('float')


#正式程序
import numpy as np
import pandas as pd
from scipy.stats import chi2, f


class preprocessing(object):
    """
    包含卡方检验、方差分析、卡方分箱、WOE编码和IV值等功能
    
    """

    #卡方检验
    def calChi2(self, x, y):
        """
        Input
        x:变量[1D]array
        y:实际标签[1D]array
        return
        chi2Value, eptFre, pValue, dfreedom
        """
        n = y.shape[0]
        xValues = np.unique(x)
        yValues = np.unique(y)
        #y的分布
        PyValues = [sum(y==yvalue)/n for yvalue in yValues]
        #生成交叉表，实际分布表和期望分布表
        realFre = np.zeros((len(xValues), len(yValues)))
        eptFre = np.copy(realFre)
        for xIdx, xvalue in enumerate(xValues):
            for yIdx,yvalue in enumerate(yValues):
                realFre[xIdx, yIdx] = sum((x==xvalue)&(y==yvalue))
                eptFre[xIdx, yIdx] = sum(x==xvalue)*PyValues[yIdx]
        #计算卡方值矩阵、卡方值、自由度、p值
        chi2Matrix = np.power((realFre-eptFre), 2)/(eptFre+1.0e-6)
        chi2Value = chi2Matrix.sum()
        dfreedom = (len(xValues)-1)*(len(yValues)-1)
        if dfreedom == 0:
            pValue = chi2.sf(chi2Value, dfreedom+1)
        else:
            pValue = chi2.sf(chi2Value, dfreedom)
        return round(chi2Value,4), round(pValue,4), dfreedom, eptFre
    
    #方差分析
    def calAnova(self, x, y):
        """
        Input
        x:变量[1D]array
        y:实际标签[1D]array
        return
        DataFrame
        """
        m = len(y)
        yValues = np.unique(y)
        n = len(yValues)
        xbar = x.mean()
        ximean = []                         #每类样本的均值列表
        xicount = []                        #每类样本的数量列表
        #计算自由度
        dfList = [n-1, m-n, m-1]            #(组间，组内，合计)
        #计算离差平方和
        ##组内
        SSList = []
        SSw = 0
        for value in yValues:
            xi = x[y==value]
            xicount.append(len(xi))         #每类的数量
            xmean = xi.mean()
            ximean.append(xmean)            #每类的均值
            SSw += np.power((xi-xmean), 2).sum()
        ##组间
        SSb = np.dot(np.power((ximean-xbar), 2), xicount)
        ##合计
        SSt = SSw + SSb
        SSList = [SSb, SSw, SSt]
        #计算均方
        MSList = [SSb/dfList[0], SSw/dfList[1]]
        #计算F值和P值
        Fvalue = MSList[0]/MSList[1]
        pValue = f.sf(Fvalue, dfList[0], dfList[1])
        #返回df
        df = pd.DataFrame(index=['组间', '组内', '合计'], columns=['自由度', '离差平方和', '均方', 'F值', 'P值'])
        df.iloc[:,0] = dfList
        df.iloc[:,1] = SSList
        df.iloc[:2,2] = MSList
        df.iloc[0,3] = Fvalue
        df.iloc[0,4] = pValue
        return df
        
    #卡方分箱
    def chibox(self, x, y, bins=4, threshold=0.01, prebox=-1):
        """
        Input
        x:变量[1D]array
        y:实际标签[1D]array
        bins:最小目标分箱数
        threshold:阈值
        prebox:初始是否先进行分箱，实例太多计算时间度大，初始值为-1，即不事先分箱
        return
        Xcates:分割点（前包后不包）
        boxcount:分箱数
        Tchi2:分箱后的卡方值
        TpValue:分箱后的P值
        """
        #对数据从小到大进行排序
        newx = np.sort(x)
        newy = y[np.argsort(x)]
        #设置初始分箱分割点
        if prebox == -1:
            Xcates = np.sort(np.unique(newx))
        else:
            Xcates = np.linspace(newx.min(), newx.max(), prebox+1)
        #设置初始分箱数、卡方值、P值
        boxcount = len(Xcates)-1
        minChi2 = np.inf; minpValue = np.inf
        while (boxcount>bins) and (minpValue>threshold):
            #内循环：计算两两的卡方值，找出最小的卡方值
            minChi2 = np.inf; minpValue = np.inf
            cutIdx = 0                                                  #初始化需要合并的点
            for cutIdx0, cutPoint0 in enumerate(Xcates):
                xi = 0; yi = 0
                if cutIdx0+2 > len(Xcates)-1:
                    break
                cutPoint1 = Xcates[cutIdx0+1]
                cutPoint2 = Xcates[cutIdx0+2]
                xi = x[(x>=cutPoint0)&(x<cutPoint2)]
                yi = y[(x>=cutPoint0)&(x<cutPoint2)]
                xi[(xi>=cutPoint0)&(xi<cutPoint1)] = cutPoint0
                xi[(xi>=cutPoint1)&(xi<cutPoint2)] = cutPoint1
                chi2, pValue, df, eptFre = self.calChi2(xi, yi)         #只有一类时，卡方值为0， 
                if chi2 < minChi2:
                    minChi2 = chi2
                    minpValue = pValue
                    cutIdx = cutIdx0 + 1
            #合并最小的卡方值的相邻分类，将需要合并的数值在Xcates中删除
            print(Xcates)
            print(cutIdx)
            Xcates = np.delete(Xcates, cutIdx)
            boxcount = len(Xcates)-1
        #计算分箱后的总体卡方值
        TXi = np.copy(x)
        for idx, value in enumerate(Xcates):
            if idx == len(Xcates)-1:
                TXi[TXi==value] = Xcates[idx-1]
            else:
                TXi[(TXi>=value)&(TXi<Xcates[idx+1])] = value
        Tchi2, TpValue, Tdf, TeptFre = self.calChi2(TXi, y)
        return Xcates, boxcount, Tchi2, TpValue
    
    #WOE编码和IV值
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        






