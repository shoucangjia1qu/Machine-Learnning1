# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:23:21 2018

@author: ecupl
"""
'''预处理总结'''
import numpy as np
import pandas as pd
import os

age = np.random.randint(0,100,size=(100,1))
pd_age = pd.DataFrame(age,columns=['age'])
#%%分箱处理
'''1、监督分箱'''
'''1-1 pd.cut()相当于为成绩设置优、良、差的有序分箱，可看作一种等量分箱'''
bins=[0,20,50,100]      #设置间隔
pd.cut(pd_age['age'],bins,labels=['A','B','C'])     #自己指定划分区间
pd.cut(pd_age['age'],4,labels=['A','B','C','D'],retbins=True)       #指定bins的数量N,实现等量均分

'''1-2 pd.qcut()也是一种有序的等频分箱'''
pd.qcut(pd_age['age'],4,retbins=True,labels=['A','B','C','D'])      #指定划分组数实现等频分箱
#retbins返还划分区间，若标识为True,则返还两部分内容，一部分是Series，另一部分为numpy.array

'''1-3 sklearn实现二分类分箱'''
from sklearn.preprocessing import Binarizer
box = Binarizer(threshold=60)               #threshold是二分类的划分界限
box.fit(age)
box.fit_transform(age)

'''1-4 sklearn实现多分类分箱'''
from sklearn.preprocessing import KBinsDiscretizer
Kbox = KBinsDiscretizer(n_bins=4,encode='ordinal',strategy='kmeans')    
#encode中onehot-dense返回密集数组，onehot返回稀疏矩阵，ordinal返回一列
#strategy中quantile表示等频分箱，uniform表示等量分箱，kmeans表示最接近中心点的分箱
Kbox.fit(age)
Kbox.fit_transform(age)
Kbox.bin_edges_             #查看分类边界

'''2、有监督分箱'''
'''2-1 卡方分箱'''
from scipy.stats import chi2
chi2.cdf(10,3)      #输入：卡方值、自由度；输出：(1-P)值
chi2.sf(10,3)       #输入：卡方值、自由度；输出：P值
chi2.ppf(0.9814338645369568,3)      #输入：(1-P)值、自由度；输出：卡方值
chi2.isf(0.01856613546304325,3)     #输入：P值、自由度；输出：卡方值
import sklearn.feature_selection as feature_selection
#2-1-1 计算卡方、P值、自由度
def CalChi2(data,labels):
    badPro = len(labels[labels==1])/len(labels)         #计算整体坏样本的概率
    dataValueCount = np.array([sum(data==v) for v in set(data)])        #计算每个值的总数
    dataBadCount = np.array([sum(np.multiply(data==v,labels==1)) for v in set(data)])       #计算每个值坏样本的个数
    EBadCount = badPro*dataValueCount       #预期的每个值坏样本的数量
    if badPro==1 or badPro==0:
        chi2value = 0
    else:
        chi2value = np.sum(np.power((dataBadCount-EBadCount),2)/EBadCount) + np.sum(np.power((EBadCount-dataBadCount),2)/(dataValueCount-EBadCount))
    df = (len(set(data))-1)
    pValue = chi2.sf(chi2value,df)
#    print(chi2value,pValue,df)
    return chi2value,pValue,df
CalChi2(data,label)
feature_selection.chi2(data.reshape(-1,1),label)    #sklearn中的卡方值和自由度都和自己算的不一致
#2-1-2 卡方分箱
def ChiBinning(data,labels,maxinterval=3,threshold=None):
    x = np.copy(data)
    x.sort()            #数据排序
    xCategory = list(set(x))         #数据值类型排序
    xCategory.sort()
    ylabels = labels[np.argsort(data)]          #y也按照data的排序来
    boxs = len(xCategory)           #分箱数
    if threshold is None:
        threshold = chi2.isf(0.05,(boxs-1))     #若未设置卡方阈值，则以0.05的置信区间作为初始阈值
    #开始循环
    while True:
        minK2 = np.inf      #设置初始最小卡方值未无穷大
        minK2Idx = np.inf   #设置最小卡方值下标为无穷大
        #开始遍历每个卡方值，找到最小卡方值
        for idx in range(boxs-1):
            subxdata = x[(x<=xCategory[idx+1])&(x>=xCategory[idx])]
            sublabels = ylabels[(x<=xCategory[idx+1])&(x>=xCategory[idx])]
            chi2value,pValue,df =CalChi2(subxdata,sublabels)
            if chi2value<minK2:
                minK2 = chi2value
                minK2Idx = idx
        #开始合并
        if minK2<threshold and boxs>maxinterval:
            #都向前合并
            x[x==xCategory[minK2Idx+1]] = xCategory[minK2Idx]
            del xCategory[minK2Idx+1]
            boxs  = len(xCategory)
        else:
            break
        print(minK2,xCategory[minK2Idx])
    return xCategory,minK2,minK2Idx

#2-1-3 根据卡方分箱结果进行分类
for idx in range(len(xCategory)):
    if idx<len(xCategory):
        data[(data>=xCategory[idx])&(data<xCategory[idx+1])] = idx
    else:
        data[data>=xCategory[idx]] = idx

#2-1-4根据分类结果计算卡方值和响应度
print(CalChi2(data,y))
'''
Chi2:1402.2546435599306;
PValue:3.193544994174465e-305;
df:2
'''
crosstab = pd.crosstab(data,y,margins=True).values
respone = crosstab/crosstab[:,-1].reshape(-1,1)
print(respone)
'''
[[0.92645208 0.07354792 1.        ]
 [0.80199253 0.19800747 1.        ]
 [0.61780105 0.38219895 1.        ]
 [0.85740981 0.14259019 1.        ]]
'''
#2-1-3 计算WOE和IV值
def CalWOEandIV(data,y):
    eps = 1e-6
    crosstab = pd.crosstab(data,y,margins=True).values
    bad = crosstab[0:-1,1]/(crosstab[-1,1]+eps)
    good = (crosstab[0:-1,0]/(crosstab[-1,0])+eps)
    woe = np.log(bad/(good+eps))
    eachIV = bad-good
    IV = np.dot(woe,eachIV)
    return woe,IV
Woe,IV = CalWOEandIV(data,y)
print(Woe,'\n',IV)
'''
WOE: [-0.73948703  0.39514035  1.313685  ] 
IV: 0.5305477313924453
'''


#%%无量纲化
'''1、 归一化(normnalization)，MaxAbsScaler也能实现缩放矩阵（保证矩阵稀疏性，取值[-1,1]）'''
age = np.random.randint(0,100,size=(100,2))
'''1-1 sklearn实现，可调用类，也可直接调用minmax_scale方法'''
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler(feature_range=(5,10))
minmax.fit(age)
mmdata=minmax.transform(age)            #得出归一化后的数据
minmax.inverse_transform(mmdata)        #还原归一化前的数据

'''1-2 numpy实现'''
def minmax(data,feature_range=(0,1)):
    minRange = feature_range[0]
    maxRange = feature_range[1]
    minData = data.min(axis=0)
    maxData = data.max(axis=0)
    mmdata = ((data-minData)/(maxData-minData))*(maxRange-minRange) + minRange
    return mmdata
mmdata2=minmax(age,feature_range=(5,10))

'''2、标准化'''
'''2-1 sklearn实现，可调用类，也可直接调用scale方法'''
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(age)
stddata = std.transform(age)            #得出标准化后的数据
stddata.mean(),stddata.std()

'''2-2 numpy实现'''
def standard(data):
    mean = data.mean(axis=0)
    var = data.std(axis=0)
    stddata = (data-mean)/var
    return stddata
stddata2 = standard(age)

'''3、正则化，可用来计算余弦相似度'''
from sklearn.preprocessing import Normalizer
norm = Normalizer(norm='l2')    #默认L2范数
normdata=norm.fit_transform(age)     
#余弦相似度(正则化后的)
cos=np.dot(normdata[0,:],normdata[1,:])
#余弦相似度(原始数据)
x1=age[0,:]
x2=age[1,:]
cos2 = np.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


#%%编码
data1=[["A","B"][np.random.randint(0,2)] for i in range(20)]
data2=[["C","A","B"][np.random.randint(0,3)] for i in range(20)]
data3=[["D","E","F"][np.random.randint(0,3)] for i in range(20)]
data=np.array([data1,data2,data3]).T

'''1、标签编码，有顺序意义，适用于一维数据'''
from sklearn.preprocessing import LabelEncoder
labelencode = LabelEncoder()
labelencode.fit(data.reshape(20))
labeldata=labelencode.transform(data)       #得到切分后的数据
labelencode.classes_                        #查看分类标签
labelencode.inverse_transform(labeldata)    #还原编码前数据

'''2、普通编码，有顺序意义的分类变量，适用于二维数据'''
from sklearn.preprocessing import OrdinalEncoder
oriencode = OrdinalEncoder(categories='auto')
oriencode.fit(data)
oridata=oriencode.transform(data)           #编码后的数据
oriencode.categories_                       #查看分类标签
oriencode.inverse_transform(oridata)        #还原编码前数据


'''3、独热编码，无顺序含义的分类变量可以稀疏化特征矩阵'''
'''3-1 sklearn实现'''
from sklearn.preprocessing import OneHotEncoder
onehotencode = OneHotEncoder(categories='auto',handle_unknown='ignore')
onehotencode.fit(data)
onehotdata=onehotencode.transform([['B','C','D']]).toarray()    #得到独热编码的数据
newFeatureName = onehotencode.get_feature_names()               #得到新的标签
onehotencode.inverse_transform(onehotdata)                      #还原编码前数据
#onehotencode.n_values()
'''参数要点：
handle_unknown:"error"和"ignore"，遇到未知类别是报错或者忽略
sparse:是否稀疏，否的话就不用toarray()
n_values:指定特征列的维度，要求连续时会自动补齐,[2,3,4]，且好像只能用于数值
categorical_features:指定需要转换的特征列，[0,2] == [True,False,True]，
                     会自动handle_unknown，且好像只能用于数值
'''

'''3-2 pandas实现'''
pd.get_dummies(pd.DataFrame(data),columns=[0,2],prefix=['数据1','data2'],
               prefix_sep=[",","_"])
'''参数要点：
drop_first:是否删除第一列，相当于N-1列
dummy_na:是否多一列显示Nan
prefix:标签分隔主名称
prefix_sep:标签分隔符
columns:指定需要编码的列
'''





