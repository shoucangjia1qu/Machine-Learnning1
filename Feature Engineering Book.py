# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:54:10 2019

@author: ecupl
"""

import os
import numpy as np
import pandas as pd
os.chdir(r"D:\mywork\test\02code_MLfeature")

#%%Chapter Two 数值型特征工程技术
#2.1 标量、向量和空间


#2.2 处理计数
##2.2.1 二值化

##2.2.2 区间量化（分箱）
#(1) 准备数据集
import json
import matplotlib.pyplot as plt
import seaborn as sns
with open(r"E:\data\Feature engineer\yelp_academic_dataset_business.json") as f:
    yelp_json = f.readlines()
yelp_List = [json.loads(row) for row in yelp_json]
yelp_df = pd.DataFrame(yelp_List)

#(2) 绘制点评数量直方图'''
sns.set_style("whitegrid")
fig, ax = plt.subplots()
#ax.set_yscale('log')                #设置y坐标为科学计数形式，有问题，取消
ax.tick_params(labelsize=14)        #设置坐标字符大小
ax.set_xlabel('Review Count', fontsize=14)
ax.set_ylabel('Occurrence', fontsize=14)
yelp_df['review_count'].hist(bins=100)
plt.show()
"""
区间量化可以将连续型数值映射为离散型数值；
当数值横跨多个数量级时，最好按照10的幂（或任何常数的幂）来进行分组；
有两种确定分箱宽度的方法：固定宽度分箱和自适应分箱。
"""

#(3) 固定宽度分箱
small_counts = np.random.randint(0, 100, 20)                        #小数据集
small_counts_box1 = np.floor_divide(small_counts, 10)               #1.1用除法向下取整，将原数据映射到相应箱中
"""---------------------------------------------------"""
small_counts_box2 = np.floor(small_counts/10)                       #1.2和上式等同
"""---------------------------------------------------"""
bins = [0,20,40,60,80,100]                                          #1.3指定bins区间，进行分箱，labels取False就是按数字顺序来
pd.cut(small_counts, bins, retbins=True, labels=False)
"""---------------------------------------------------"""
pd.cut(small_counts, bins=5, labels=['A','B','C','D','E'])          #1.4指定bins数量，进行等距分箱
"""---------------------------------------------------"""
large_counts = [296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 11495, 91897,
                44, 28, 7971, 926, 122, 22222]                      #横跨多个数量级的数据集
large_counts_box = np.floor(np.log10(large_counts))                 #1.5对数据求幂后，再向下取整

#(4) 分位数分箱(自适应分箱)，实质是一种等量分箱
deciles = yelp_df['review_count'].quantile([.1, .2, .3, .4, .5, .6, .7, 
                 .8, .9, 1])                                        #计算十分位数，并画图
sns.set_style("whitegrid")
fig, ax = plt.subplots()
yelp_df['review_count'].hist(bins=100)
for pos in deciles:
    handle = plt.axvline(x=pos, color='r')
ax.set_xscale('log')
#ax.set_yscale('log')
ax.tick_params(labelsize=14)
ax.set_xlabel('Review Count', fontsize=14)
ax.set_ylabel('Occurrence', fontsize=14)
plt.show()
"""---------------------------------------------------"""
pd.qcut(yelp_df.review_count, q=4, labels=False)                    #2.1分位数分箱，用qcut，指定好q就行
yelp_df.review_count.quantile([0.25, 0.5, 0.75])                    #计算实际的分位数的值
pd.qcut(yelp_df.review_count, q=4, labels=False, retbins=True)[1]   #用qcut取出元组的第二个元素也是实际的分位数值
"""---------------------------------------------------"""
pd.qcut(yelp_df.review_count, q=10, labels=False, \
        duplicates='drop', retbins=True)                            #2.2分位数分箱，当分位数的值有重复时，需要输入duplicate参数


#2.3 对数变换
##2.3.1 画图
yelp_df['log_review_count'] = np.log(yelp_df.review_count+1)        #对数变换
#(1) 自己的方法
plt.subplot(2,1,1)
yelp_df['log_review_count'].hist(bins=100)
plt.subplot(2,1,2)
yelp_df['review_count'].hist(bins=100)

#(2) 书上的方法
fig, (ax1, ax2) = plt.subplots(2,1)
yelp_df['review_count'].hist(ax=ax1, bins=100)                      #转换前分布图
ax1.tick_params(labelsize=14)
ax1.set_xlabel('review_count', fontsize=14)
ax1.set_ylabel('Occurrence', fontsize=14)
yelp_df['log_review_count'].hist(ax= ax2, bins=100)                 #转换后分布图
ax2.tick_params(labelsize=14)
ax2.set_xlabel('log10(review_count)', fontsize=14)
ax2.set_ylabel('Occurrence', fontsize=14)

##2.3.2 再画图
news_df = pd.read_csv(r"E:\data\Feature engineer\OnlineNewsPopularity.csv", delimiter=",")
news_df['log_n_tokens_content'] = np.log(news_df[' n_tokens_content']+1)
#(1) 自己的方法
plt.subplot(2,1,1)
news_df[' n_tokens_content'].hist(bins=100)                         #转换前分布图
plt.tick_params(labelsize=14)
plt.xlabel('Number of Words in Article', fontsize=14)
plt.ylabel('Number of Articles', fontsize=12)
plt.subplot(2,1,2)
news_df['log_n_tokens_content'].hist(bins=100)                      #转换后分布图
plt.tick_params(labelsize=14)
plt.xlabel('Log of Number of Words', fontsize=14)
plt.ylabel('Number of Articles', fontsize=12)

##2.3.3 实战1：使用对数变换后的Yelp点评数量预测商家的平均评分
#(1) 导入包
import json
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

#(2) 导入数据
with open(r"E:\data\Feature engineer\yelp_academic_dataset_business.json") as f:
    yelp_json = f.readlines()
yelp_List = [json.loads(row) for row in yelp_json]
yelp_df = pd.DataFrame(yelp_List)

#(3) 对数转换
yelp_df['log_review_count'] = np.log10(yelp_df.review_count + 1)

#(4) 单变量线性回归
clf_ori = linear_model.LinearRegression()                           #原始数据集的线性模型
scores_ori = cross_val_score(clf_ori, yelp_df[['review_count']], yelp_df['stars'], cv=10)
print("R-squared score without log transform:%.5f (+/-%.5f)"%(scores_ori.mean(), scores_ori.std()*2))
"""---------------------------------------------------"""
clf_log = linear_model.LinearRegression()                           #对数转换后数据集的线性模型
scores_log = cross_val_score(clf_log, yelp_df[['log_review_count']], yelp_df['stars'], cv=10)
print("R-squared score without log transform:%.5f (+/-%.5f)"%(scores_log.mean(), scores_log.std()*2))

##2.3.4 实战2：使用对数变换后的新闻数据预测分享数
#(1) 数据准备
news_df = pd.read_csv(r"E:\data\Feature engineer\OnlineNewsPopularity.csv", delimiter=",")

#(2) 对数转换
news_df['log_n_tokens_content'] = np.log10(news_df[' n_tokens_content'] + 1)

#(3) 单变量线性回归
clf_ori = linear_model.LinearRegression()                           #新闻原数据集的线性模型
scores_ori = cross_val_score(clf_ori, news_df[[' n_tokens_content']], news_df[' shares'], cv=10)
print("R-squared score without log transform:%.5f (+/-%.5f)"%(scores_ori.mean(), scores_ori.std()*2))
"""---------------------------------------------------"""
clf_log = linear_model.LinearRegression()                           #新闻对数转换后数据集的线性模型
scores_log = cross_val_score(clf_log, news_df[['log_n_tokens_content']], news_df[' shares'], cv=10)
print("R-squared score without log transform:%.5f (+/-%.5f)"%(scores_log.mean(), scores_log.std()*2))

##2.3.5 输入输出散点图可视化：新闻流行度数据
fig2, (ax1, ax2) = plt.subplots(2,1)
ax1.scatter(news_df[' n_tokens_content'], news_df[' shares'])       #转换前散点图
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Number of Words in Article', fontsize=14)
ax1.set_ylabel('Number of Shares', fontsize=14)
"""对数变换将Y值异常巨大的文章，更多地拉向了X轴的右侧，
   为线性模型在输入特征空间的低值端争取了更多的“呼吸空间”"""
ax2.scatter(news_df['log_n_tokens_content'], news_df[' shares'])    #转换后散点图
ax2.tick_params(labelsize=14)
ax2.set_xlabel('Number of Words in Article', fontsize=14)
ax2.set_ylabel('Number of Shares', fontsize=14)
plt.show()

##2.3.6 输入输出散点图可视化：商家点评数据
fig2, (ax1, ax2) = plt.subplots(2,1)
ax1.scatter(yelp_df['review_count'], yelp_df['stars'])              #转换前散点图
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Review Count', fontsize=14)
ax1.set_ylabel('Average Star Rating', fontsize=14)
ax2.scatter(yelp_df['log_review_count'], yelp_df['stars'])          #转换后散点图
ax2.tick_params(labelsize=14)
ax2.set_xlabel('Review Count', fontsize=14)
ax2.set_ylabel('Average Star Rating', fontsize=14)
plt.show()

##2.3.7 指数变换
"""Box-Cox变换"""
#(1) 导入包
from scipy import stats

#(2) 指数变换
rc_log = stats.boxcox(yelp_df['review_count'], lmbda=0)             #lambda=0，相当于是对数变换
"""---------------------------------------------------"""
rc_bc, bc_params = stats.boxcox(yelp_df['review_count'])            #默认情况下，会找出使得输出最接近于正态分布的lambda参数
print(bc_params)

#(3) 分布图
fig2, (ax1, ax2, ax3) = plt.subplots(3,1)
yelp_df['review_count'].hist(ax=ax1, bins=100)                      #原始数据分布图
plt.subplot(3,1,2)
plt.hist(rc_log, bins=100)                                          #对数转换后的分布图，lambda=0
plt.subplot(3,1,3)
plt.hist(rc_bc, bins=100)                                           #指数转换后的分布图，lambda为最优
plt.show()

#(4) 概率图
fig2, (ax1, ax2, ax3) = plt.subplots(3,1)
prob1 = stats.probplot(yelp_df['review_count'], dist=stats.norm, plot=ax1)
prob1 = stats.probplot(rc_log, dist=stats.norm, plot=ax2)
prob1 = stats.probplot(rc_bc, dist=stats.norm, plot=ax3)


#2.4 特征缩放/归一化
#(1) 导入包
import pandas as pd
import sklearn.preprocessing as preproc

#(2) 导入数据
news_df = pd.read_csv(r"E:\data\Feature engineer\OnlineNewsPopularity.csv", delimiter=",")

#(3) 进行特征缩放
news_df[' n_tokens_content'].as_matrix()                                    #原数据
"""---------------------------------------------------"""
news_df['minmax'] = preproc.minmax_scale(news_df[[' n_tokens_content']])    #min-max缩放
"""---------------------------------------------------"""
news_df['standardized'] = preproc.StandardScaler().\
fit_transform(news_df[[' n_tokens_content']])                               #中心标准化，均值为0，标准差为1  
"""---------------------------------------------------"""
news_df['l2_normalized'] = preproc.normalize(news_df[[' n_tokens_content']], \
       axis=0)                                                              #归一化

#(4) 可视化，查看分布图
fig2, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
news_df[' n_tokens_content'].hist(ax=ax1, bins=100)
news_df['minmax'].hist(ax=ax2, bins=100)
news_df['standardized'].hist(ax=ax3, bins=100)
news_df['l2_normalized'].hist(ax=ax4, bins=100)
plt.show()
"""原始及缩放后的单词数量，只有x轴的尺度发生了变化，分布形状保持不变；
   当一组输入特征的尺度相差很大时，就需要进行特征缩放；
   例如：一个人气网站日访问量可能是几十万次，而实际购买行为可能只有几千次，
   如果这两个特征都在模型中，就需要先平衡下尺度，不然不稳定。"""


#2.5 交互特征，生成多项式
#(1) 导入包
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preproc
#(2) 导入数据
news_df = pd.read_csv(r"E:\data\Feature engineer\OnlineNewsPopularity.csv", delimiter=",")
features = [' n_tokens_title', ' n_tokens_content',
       ' n_unique_tokens', ' n_non_stop_words', ' n_non_stop_unique_tokens',
       ' num_hrefs', ' num_self_hrefs', ' num_imgs', ' num_videos',
       ' average_token_length', ' num_keywords', ' data_channel_is_lifestyle',
       ' data_channel_is_entertainment', ' data_channel_is_bus',
       ' data_channel_is_socmed', ' data_channel_is_tech',
       ' data_channel_is_world']
X = news_df[features]
y = news_df[[' shares']]
#(3) 创建交互特征对，跳过固定偏移项
X2 = preproc.PolynomialFeatures(include_bias=False).fit_transform(X)
print(X2.shape)
#(4) 为两个特征集创建训练集和测试集
X1_train, X1_test, X2_train, X2_test , y_train, y_test = \
    train_test_split(X, X2, y, test_size=0.3, random_state=123)
#(5) 模型训练和R2评分函数
def evaluate_feaature(X_train, X_test, y_train, y_test):
    """未添加截距列，故在模型中需要fit_intercept"""
    model = linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    r2_score = model.score(X_test, y_test)
    return (model, r2_score)
#(6) 比较R2分数
m1, r1 = evaluate_feaature(X1_train, X1_test, y_train, y_test)
m2, r2 = evaluate_feaature(X2_train, X2_test, y_train, y_test)
print("R2 score with singleton features: %0.5f" % r1)
print("R2 score with pairwise features: %0.10f" % r2)


#%%Chapter Three 文本数据：扁平化、过滤和分块
#3.1 元素袋：将自然文本转换为扁平向量


