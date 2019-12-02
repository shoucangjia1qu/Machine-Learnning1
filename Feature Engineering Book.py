# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:54:10 2019

@author: ecupl
"""

import os
import numpy as np
import pandas as pd
os.chdir(r"D:\mywork\test")

#处理计数
##二值化

##区间量化（分箱）
'''1、准备数据集'''
import json
with open(r"E:\data\Feature engineer\yelp_academic_dataset_business.json") as f:
    yelp_json = f.readlines()
yelp_List = [json.loads(row) for row in yelp_json]
yelp_df = pd.DataFrame(yelp_List)
'''2、绘制点评数量直方图'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
fig, ax = plt.subplots()
#ax.set_yscale('log')                #设置y坐标为科学计数形式，有问题，取消
ax.tick_params(labelsize=14)        #设置坐标字符大小
ax.set_xlabel('Review Count', fontsize=14)
ax.set_ylabel('Occurrence', fontsize=14)
yeld_df['review_count'].hist(bins=100)
plot.show()
'''
Q：点评数据横跨了若干个数量级。
可进行区间量化
'''
###区间量化方法
'''1、固定宽度分箱'''
'''一般指定宽度进行分箱，如果横跨多个数量级，最好按照10的幂来进行分组'''
small_counts = np.random.randint(0, 100, 20)                        #小数据集
small_counts_box1 = np.floor_divide(small_counts, 10)               #1.1用除法向下取整，将原数据映射到相应箱中

small_counts_box2 = np.floor(small_counts/10)                       #1.2和上式等同

bins = [0,20,40,60,80,100]                                          #1.3指定bins区间，进行分箱，labels取False就是按数字顺序来
pd.cut(small_counts, bins, retbins=True, labels=False)

pd.cut(small_counts, bins=5, labels=['A','B','C','D','E'])          #1.4指定bins数量，进行等距分箱

large_counts = [296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 11495, 91897,
                44, 28, 7971, 926, 122, 22222]                      #横跨多个数量级的数据集
large_counts_box = np.floor(np.log10(large_counts))                 #1.5对数据求幂后，再向下取整

'''2、分位数分箱，实质是一种等量分箱'''
'''根据数据的分布特点，进行自适应的箱体定位'''
deciles = yelp_df['review_count'].quantile([.1, .2, .3, .4, .5, .6, .7, 
                 .8, .9, 1])                                                  #计算十分位数，并画图
sns.set_style("whitegrid")
fig, ax = plt.subplots()
ax.set_xscale('log')
#ax.set_yscale('log')
ax.tick_params(labelsize=14)
ax.set_xlabel('Review Count', fontsize=14)
ax.set_ylabel('Occurrence', fontsize=14)
yeld_df['review_count'].hist(bins=100)
for pos in deciles:
    handle = plt.axvline(x=pos, color='r')
plot.show()

pd.qcut(yelp_df.review_count, q=4, labels=False)                    #2.1分位数分箱，用qcut，指定好q就行
yelp_df.review_count.quantile([0.25, 0.5, 0.75])                    #计算实际的分位数的值
pd.qcut(yelp_df.review_count, q=4, labels=False, retbins=True)[1]   #用qcut取出元组的第二个元素也是实际的分位数值


##对数变换
'''1、画图（变换前和变换后）,Yelp商家点评数据集'''
yelp_df['log_review_count'] = np.log(yelp_df.review_count+1)        #对数变换
'''自己的方法'''
plt.subplot(2,1,1)
yelp_df['log_review_count'].hist(bins=100)
plt.subplot(2,1,2)
yelp_df['review_count'].hist(bins=100)
'''书上的方法'''
fig, (ax1, ax2) = plt.subplots(2,1)
yelp_df['review_count'].hist(ax=ax1, bins=100)
ax1.tick_params(labelsize=14)
ax1.set_xlabel('review_count', fontsize=14)
ax1.set_ylabel('Occurrence', fontsize=14)

yelp_df['log_review_count'].hist(ax= ax2, bins=100)
ax2.tick_params(labelsize=14)
ax2.set_xlabel('log10(review_count)', fontsize=14)
ax2.set_ylabel('Occurrence', fontsize=14)

'''2、再画图，在线新闻数据集'''
news_df = pd.read_csv(r"E:\data\Feature engineer\OnlineNewsPopularity.csv", delimiter=",")
news_df['log_n_tokens_content'] = np.log(news_df[' n_tokens_content']+1)

plt.subplot(2,1,1)
plt.tick_params(labelsize=14)
plt.xlabel('Number of Words in Article', fontsize=14)
plt.ylabel('Number of Articles', fontsize=12)
news_df[' n_tokens_content'].hist(bins=100)

plt.subplot(2,1,2)
plt.tick_params(labelsize=14)
plt.xlabel('Log of Number of Words', fontsize=14)
plt.ylabel('Number of Articles', fontsize=12)
news_df['log_n_tokens_content'].hist(bins=100)

'''实战1：使用对数变换后的Yelp点评数量预测商家的平均评分'''
import json
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

with open(r"E:\data\Feature engineer\yelp_academic_dataset_business.json") as f:
    yelp_json = f.readlines()
yelp_List = [json.loads(row) for row in yelp_json]
yelp_df = pd.DataFrame(yelp_List)
yelp_df['log_review_count'] = np.log(yelp_df.review_count + 1)

clf_ori = linear_model.LinearRegression()                   #原始数据集的线性模型
scores_ori = cross_val_score(clf_ori, yelp_df[['review_count']], yelp_df['stars'], cv=10)
print("R-squared score without log transform:%.5f (+/-%.5f)"%(scores_ori.mean(), scores_ori.std()*2))

clf_log = linear_model.LinearRegression()                   #对数转换后数据集的线性模型
scores_log = cross_val_score(clf_log, yelp_df[['log_review_count']], yelp_df['stars'], cv=10)
print("R-squared score without log transform:%.5f (+/-%.5f)"%(scores_log.mean(), scores_log.std()*2))

'''实战2：使用对数变换后的新闻数据预测分享数'''
news_df['log_n_tokens_content'] = np.log(news_df[' n_tokens_content'] + 1)

clf_ori = linear_model.LinearRegression()                   #新闻原数据集的线性模型
scores_ori = cross_val_score(clf_ori, news_df[[' n_tokens_content']], news_df[' shares'], cv=10)
print("R-squared score without log transform:%.5f (+/-%.5f)"%(scores_ori.mean(), scores_ori.std()*2))

clf_log = linear_model.LinearRegression()                   #新闻对数转换后数据集的线性模型
scores_log = cross_val_score(clf_log, news_df[['log_n_tokens_content']], news_df[' shares'], cv=10)
print("R-squared score without log transform:%.5f (+/-%.5f)"%(scores_log.mean(), scores_log.std()*2))





















