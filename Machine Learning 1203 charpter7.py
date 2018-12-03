# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:04:34 2018

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

'''读取数据集'''
os.chdir(r"D:\mywork\test\ML")
with open("regdataset.txt","r") as f:
    content = f.readlines()
dataList = [[float(y) for y in x.split()] for x in content]
dataSet = np.array(dataList)

'''计算斜率和截距'''
n = len(dataSet)
x = dataSet[:,0]
y = dataSet[:,1]
xmean = np.mean(x)
ymean = np.mean(y)
a = (np.dot(x,y)-n*xmean*ymean)/(sum(np.power(x,2))-n*xmean**2)
b = ymean - a*xmean

'''画图展示'''
preY = a*x+b
plt.figure()
plt.scatter(x,y)
plt.legend(['plot'],loc=2)
plt.plot(x,preY,c='r',linewidth=3)
plt.xlabel("X")
plt.ylabel("Y")
#plt.legend(['preY'],loc=4)
plt.show()


