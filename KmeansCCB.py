# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:43:08 2018

@author: ZWD
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy, os
from sklearn import preprocessing
from sklearn.decomposition import PCA
from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
from sklearn.cluster import KMeans
from sklearn import metrics

os.chdir(r"D:\mywork\test\ML_CCB")
##############################
#                            #
#     1、定义函数、算法       #
#                            #
##############################

'''1-1聚类个数评价：轮廓系数'''
def LK(train,labels):
    LK = []
    m = 0
    for data in train:
        n=0
        a = 0
        b = dict()
        avalue = 0
        bvalue = 0
        for subdata in train: 
            if m==n:
                n += 1
                continue
            if Labels[m] == Labels[n]:
                a += KM.eDist(data,subdata)
            else:
                if Labels[n] not in b.keys():
                    b[Labels[n]] = 0
                b[Labels[n]] += KM.eDist(data,subdata)
            n += 1
        '''a是点到本簇中其他点的平均距离'''
        avalue = (a/(len(np.nonzero(Labels==Labels[m])[0])-1))
        '''b是点到其他簇中其他点的平均距离的最小值'''
        bvalue = np.min([value/len(np.nonzero(Labels==la)[0]) for la,value in b.items()])
        LK.append((bvalue-avalue)/max(bvalue,avalue))
        m += 1
    LKratio = np.mean(LK)
    return(LKratio)

'''1-2极值处理函数'''
def maxNum(data):
    std=np.std(data)
    mean=np.mean(data)
    vmax=5*std+mean
    data[data>vmax]=vmax
    return data

'''1-3数据标准化'''
def normlize(data):
    m,n = data.shape
    for col in n:
        for row in m:
            normdata[row,col] = (data[row,col]-data[:,col].mean())/data[:,col].std()
    return normdata


##############################
#                            #
#     2、数据处理             #
#                            #
##############################
'''2-1读入数据，区分连续变量和分类标量'''
data = pd.read_excel("CLUSTER_PRIVATE.xlsx")
cateList = data.columns.tolist()[1:9]
cateList.extend(['address','id'])                   #分类变量
conList = data.columns.tolist()
for x in cateList:
    conList.remove(x)                               #连续变量
cateData = data[cateList]
conData = data[conList]
'''2-2预处理(填缺、极大值处理)'''
conData.age.fillna(conData.age.median(), inplace=True)        #年龄中位数填缺
conData.iloc[:,1:] = conData.iloc[:,1:].fillna(0)             #其他行为用0填缺
for i in conList:
    conData[i] = maxNum(conData[i])             #极值处理
print(conData.describe().T)                     #查看基本统计情况
corrMa = conData.iloc[:,1:].corr()              #相关系数矩阵
'''2-3发现有不少变量相关性较高，先梳理出来'''
PcaList = conList[1:conList.index('outside_cnum')+1]
PcaList.extend(conList[-8:-1])
PcaData = conData[PcaList]
'''2-4业务上先按类将客户消费行为分类
包括：餐饮类、休闲娱乐类、固定资产类、出行类、家用生活类、公共事业类、金融服务类、其他类'''
comsumeData = pd.DataFrame()
comsumeData['lunch_money'] = conData['lunch_cmoney']+conData['food_cmoney']
comsumeData['lunch_times'] = conData['lunch_cnum']+conData['food_cnum']
comsumeData['play_money'] = conData['play_cmoney']+conData['golf_cmoney']+conData['heal_cmoney']+\
                            conData['travel_cmoney']+conData['carrent_cmoney']
comsumeData['play_times'] = conData['play_cnum']+conData['golf_cnum']+conData['heal_cnum']+\
                            conData['travel_cnum']+conData['carrent_cnum']
comsumeData['asset_money'] = conData['flat_cmoney']+conData['car_cmoney']
comsumeData['asset_times'] = conData['flat_cnum']+conData['car_cnum']
comsumeData['travel_money'] = conData['air_cmoney']+conData['oil_cmoney']+conData['rail_cmoney']+\
                              conData['carbt_cmoney']+conData['airpt_cmoney']+conData['carsev_cmoney']
comsumeData['travel_times'] = conData['air_cnum']+conData['oil_cnum']+conData['rail_cnum']+\
                              conData['carbt_cnum']+conData['airpt_cnum']+conData['carsev_cnum']
comsumeData['family_money'] = conData['jewey_cmoney']+conData['market_cmoney']+conData['jiadian_cmoney']+\
                              conData['store_cmoney']+conData['com_cmoney']+conData['cloth_cmoney']+\
                              conData['tel_cmoney']+conData['cgy_cmoney']+conData['animal_cmoney']+\
                              conData['pay_cmoney']+conData['mgt_cmoney']+conData['pref_cmoney']
comsumeData['family_times'] = conData['jewey_cnum']+conData['market_cnum']+conData['jiadian_cnum']+\
                              conData['store_cnum']+conData['com_cnum']+conData['cloth_cnum']+\
                              conData['tel_cnum']+conData['cgy_cnum']+conData['animal_cnum']+\
                              conData['pay_cnum']+conData['mgt_cnum']+conData['pref_cnum']
comsumeData['public_money'] = conData['hosp_cmoney']+conData['sch_cmoney']+conData['gov_cmoney']+conData['tax_cmoney']
comsumeData['public_times'] = conData['hosp_cnum']+conData['sch_cnum']+conData['gov_cnum']+conData['tax_cnum']
comsumeData['fin_money'] = conData['ins_cmoney']+conData['fin_cmoney']
comsumeData['fin_times'] = conData['ins_cnum']+conData['fin_cnum']
comsumeData['other_money'] = conData['hotel_cmoney']+conData['pifa_cmoney']+conData['othersev_cmoney']+conData['other_cmoney']
comsumeData['other_times'] = conData['hotel_cnum']+conData['pifa_cnum']+conData['othersev_cnum']+conData['other_cnum']
comsumeData_scale = preprocessing.scale(comsumeData)          #数据中心标准化
comsumeCorrMa = comsumeData.corr()
'''2-5对16个消费类变量也进行变量聚类(聚成8个)，降维'''
pca=PCA(n_components=8)
newData=pca.fit(comsumeData_scale)
print(pca.explained_variance_)
'''
[ 4.55020824  1.54386808  1.2895492   1.17837481  1.00672056  0.96910209
  0.94731602  0.85934027 ]'''
print(pca.explained_variance_ratio_)
'''
[ 0.2843533   0.09647998  0.08058699  0.07363944  0.06291235  0.06056149
  0.05920002  0.05370221 ]
'''
PCA_factor = pd.DataFrame(pca.components_).T
'''PCA因子有点杂乱，进行FCA'''
fa = FactorAnalysis.load_data_samples(
        comsumeData_scale,
        preproc_demean=True,
        preproc_scale=True
        )
fa.extract_components()
fa.find_comps_to_retain(method='top_n',num_keep=8)
fa.rotate_components(method='varimax')
fa_plotting.graph_summary(fa)
FCA_factor = pd.DataFrame(fa.comps["rot"])
cnsmpData = pd.DataFrame(np.dot(comsumeData_scale,FCA_factor))
cnsmpData = cnsmpData.rename(columns={0: "lunch&family_times", 1: "asset_cnsmp", 2: "public_cnsmp", 3:"fin_cnsmp", 4:"other_cnsmp", \
                                  5:"travel_cnsmp", 6:"play_cnsmp", 7:"lunch&family_money"})

'''2-6对相关性高的变量进行处理,PCA主成分分析'''
PcaCorrMa = PcaData.corr()
PcaData.drop(['crcrd_totmoney', 'crcrd_totnum', 'dbcrd_totmoney', 'dbcrd_totnum'],axis=1,inplace=True)
PcaData_scale = preprocessing.scale(PcaData)
pca=PCA(n_components=21)
newData=pca.fit(PcaData_scale)
print(pca.explained_variance_)
'''
[ 4.44714283  2.22045512  2.03470115  1.64845957  1.56942443  1.42609679
  1.36034065  1.0094452   0.86141772  0.78580677  0.77398548  0.63857826
  0.39624092  0.37043049  0.35589455  0.31758863  0.27880595  0.21271549
  0.12904495  0.08929104  0.07669778]
'''
print(pca.explained_variance_ratio_)
'''
[ 0.21174286  0.10572305  0.0968787   0.07848849  0.07472537  0.06790108
  0.06477022  0.04806295  0.04101488  0.0374148   0.03685195  0.03040478
  0.01886631  0.01763739  0.01694529  0.01512142  0.01327485  0.01012807
  0.00614425  0.00425144  0.00365183]
'''
'''选择9个主成分'''
pca=PCA(n_components=9)
newData=pca.fit(PcaData_scale)
PCA_factor = pd.DataFrame(pca.components_).T
'''主成分不太好归纳，重新做因子转换'''
fa = FactorAnalysis.load_data_samples(
        PcaData_scale,
        preproc_demean=True,
        preproc_scale=True
        )
fa.extract_components()
fa.find_comps_to_retain(method='top_n',num_keep=9)
fa.rotate_components(method='varimax')
fa_plotting.graph_summary(fa)
FCA_factor = pd.DataFrame(fa.comps["rot"])
'''生成降维后的数据，并根据FCA因子改变列名'''
newpcaData = pd.DataFrame(np.dot(PcaData_scale,FCA_factor))
newpcaData = pcaData.rename(columns={0: "Fpay&Online_consume", 1: "tentbonus", 2: "crcrd_huankuan", 3:"AUM", 4:"bank&auto_trade", \
                                  5:"loan_huandai", 6:"oversea_consume", 7:"pos_trade", 8:"elec_trade"})

'''2-10合成数据集'''
FinalData = pd.DataFrame()
FinalData = data[cateList]
FinalData['age'] = preprocessing.scale(conData.age)             #年龄也进行中心标准化
FinalData = FinalData.join(newpcaData)
FinalData = FinalData.join(cnsmpData)

##############################
#                            #
#     3、开始聚类             #
#                            #
##############################

'''3-1正式程序'''
train = FinalData.loc[:,'age':]
trainTitle = train.columns.tolist()
train = np.array(train)
'''3-2用sklearn封装好的算法'''
kmeans = KMeans(n_clusters=8)
result=kmeans.fit(train)
labels = result.labels_
'''记录每个标签类别的数量'''
labelCount = dict()
for i in set(labels):
    labelCount[i] = len(labels[labels == i])
'''3-3轮廓系数验证，最终选择8类'''
print(metrics.silhouette_score(train, labels, metric='euclidean'))
'''
3 0.57431112133
4 0.525698608286
5 0.535746151298
6 0.49241004761
7 0.446090559268
8 0.539479920551
9 0.483630885352
10 0.429374131464
11 0.417886033048
12 0.409798308227
13 0.164715173315
14 0.391274980922
15 0.392947358311
16 0.382208321408
17 0.383790615587
18 0.187900931917
19 0.194410020544
'''

##############################
#                            #
#     4、结果解读             #
#                            #
##############################
'''4-1先用决策树拟合'''
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=20, min_samples_leaf=20, random_state=1234) 
clf.fit(train, labels)
'''4-2展示决策树分类效果'''
import pydotplus
from IPython.display import Image
import sklearn.tree as tree
dot_data = tree.export_graphviz(clf, 
                                out_file=None, 
                                feature_names=trainTitle,  
                                class_names=['0','1','2','3','4','5','6','7'],
                                filled=True) 
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 

FinalData['labels'] = pd.DataFrame(labels)
FinalData.to_excel("FinalData1113_withResult.xlsx")                #保存


##############################
#                            #
#  5、业务层面寻找特征        #
#                            #
##############################













