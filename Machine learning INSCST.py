# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 10:41:15 2018

@author: ecupl
"""

import os
os.chdir("D:\\mywork\\test\\ML_CCB")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
'''设置行列显示数'''
pd.set_option('max_columns',100)
pd.set_option('max_rows',100)
#导入数据
data_ori=pd.read_excel("INS_CST.xlsx")    #读取数据
'''data_ori=pd.read_csv("INS_CST.csv") '''   
data_ori['data']=data_ori['data'].astype(pd.datetime)    #转换成日期格式
data_ori=data_ori.sort_values(by='data')
data_ori.head(10)
##################基本统计分析######################
xn,yn=data_ori.shape    #行和列数
data_ori.describe(include='all').T      #总览
'''查看每个字段0和1的平均值进行对比,初步判断可以删掉businessloan和carloan'''
data_ori.groupby('result').mean()
'''样本响应度14.26%'''
data_ori.result.value_counts()
'''gender交叉占比表'''
tab_gender=pd.crosstab(index=data_ori.result,columns=data_ori.gender)
tab_gender.div(tab_gender.sum(0),axis=1)
'''edu交叉占比表,缺失50%，响应度9%，可删除'''
tab_edu=pd.crosstab(index=data_ori.edu,columns=data_ori.result)
tab_edu.div(tab_edu.sum(1),axis=0)
'''marriage交叉占比表,缺失37.5%，响应度7%'''
tab_marriage=pd.crosstab(index=data_ori.marriage,columns=data_ori.result)
tab_marriage.div(tab_marriage.sum(1),axis=0)
'''rank交叉占比表,缺失15%，响应度6%'''
tab_rank=pd.crosstab(index=data_ori['rank'],columns=data_ori.result)
tab_rank.div(tab_rank.sum(1),axis=0)
'''familynum交叉占比表,缺失86%，响应度13.4%，应删除'''
tab_familynum=pd.crosstab(index=data_ori.familynum,columns=data_ori.result)
tab_familynum.div(tab_familynum.sum(1),axis=0)
'''其他分类标量的交叉表与响应度统计'''
list1=['crmmanage','flatloan', 'businessloan',
       'consumeloan', 'otherloan', 'foreigncurrency', 'ctssign', 'creditstage',
       'telebank', 'messbank', 'netbank', 'mobilebank', 'wechatbank',
       'fastloan', 'carloan', 'gongjijin', 'daifa']
for i in list1:
    tab=pd.crosstab(index=data_ori['{}'.format(i)],columns=data_ori.result)
    print(tab.div(tab.sum(1),axis=0))
    print("------------------")
'''
1、响应度上daifa,gongjijin,telebank在响应度上差异低，且在14%上下，应重点关注
2、carloan,businessloan无响应度，应删除
'''
##############初步清洗#################
data_ori.columns
'''删除无响应度变量carloan,businessloan
   删除高缺失变量
   删除无关变量'''
data_clear1=data_ori.drop(['edu','familynum','carloan','businessloan'],axis=1)
'''填补缺失值'''
data_clear1['age']=data_clear1['age'].fillna(np.mean(data_ori.age))    #用中位数填补
data_clear1.marriage=data_clear1.marriage.fillna('@000')    #填补婚姻缺失值,缺失较多，且响应差异较大，故自成一类
data_clear1.marriage.replace(to_replace='@',value='@000',inplace=True)     #替换婚姻状态
data_clear1.gender=data_clear1.gender.fillna(99)    #填补性别缺失值,等会要删掉
data_clear1['rank']=data_clear1['rank'].fillna(999)    #填补等级缺失值，缺失较多，且响应差异较大，故自成一类
data_clear1[[ 'alltrade', 'banktrade', 'autotrade', 'electrade', 'postrade',
       'othertrade', 'money_dbcrd', 'times_dbcrd', 'emoney_dbcrd',
       'etimes_dbcrd', 'money_crcrd', 'times_crcrd', 'emoney_crcrd',
       'etimes_crcrd', 'money_fp', 'times_fp', 'money_albb', 'times_albb',
       'money_tent', 'times_tent']]=data_clear1[[ 'alltrade', 'banktrade', 'autotrade', 'electrade', 'postrade',
       'othertrade', 'money_dbcrd', 'times_dbcrd', 'emoney_dbcrd',
       'etimes_dbcrd', 'money_crcrd', 'times_crcrd', 'emoney_crcrd',
       'etimes_crcrd', 'money_fp', 'times_fp', 'money_albb', 'times_albb',
       'money_tent', 'times_tent']].fillna(0)    #交易信息、消费信息、第三方信息,未取到，就是0
data_clear1[['flatloan',
       'consumeloan', 'otherloan', 'foreigncurrency', 'ctssign', 'creditstage',
       'telebank', 'messbank', 'netbank', 'mobilebank', 'wechatbank',
       'fastloan',  'gongjijin', 'daifa', 'avgaum', 'cash',
       'cardmoney', 'loan', 'cts', 'allinvest', 'fundaum', 'goldaum',
       'bondaum', 'bankinvestaum', 'insaum', 'trustaum', 'otheraum',
]]=data_clear1[['flatloan', 
       'consumeloan', 'otherloan', 'foreigncurrency', 'ctssign', 'creditstage',
       'telebank', 'messbank', 'netbank', 'mobilebank', 'wechatbank',
       'fastloan', 'gongjijin', 'daifa', 'avgaum', 'cash',
       'cardmoney', 'loan', 'cts', 'allinvest', 'fundaum', 'goldaum',
       'bondaum', 'bankinvestaum', 'insaum', 'trustaum', 'otheraum',
]].fillna(0)    #产品覆盖信息、aum信息，未取到，就是0
data_clear1=data_clear1[data_clear1['gender']!=99]      #删除性别为99的样本
data_clear1.marriage=data_clear1.marriage.apply(lambda x : str(x)[1:])      #婚姻等级选取
data_clear1.describe(include='all').T
xnew,ynew=data_clear1.shape
'''将样本量和测试量区分'''
data_sample=data_clear1[data_clear1['result'].isin (['0','1'])]
data_sample.drop('newid',axis=1,inplace=True)
data_target=data_clear1[~data_clear1['result'].isin (['0','1'])]
#############分类变量筛选#############
data_sample.reset_index(drop=True,inplace=True)     #重设index
'''分类变量进行卡方检验'''
import sklearn.feature_selection as feature_selection
feature_selection.chi2(data_sample[['num','marriage','gender',
                                    'crmmanage','rank','flatloan',
                                    'consumeloan','otherloan','foreigncurrency',
                                    'ctssign','creditstage','telebank',
                                    'messbank','netbank','mobilebank',
                                    'wechatbank','fastloan','gongjijin','daifa']],data_sample['result'])
'''gender、mobile不显著，可删除'''
'''向前法'''
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            aic = smf.glm(
                formula=formula, data=data, 
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('aic is {},continuing!'.format(current_score))
        else:        
            print ('forward selection over!')
            break
            
    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    model = smf.glm(
        formula=formula, data=data, 
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return(model)
candidates = ['result','num','marriage',
                                    'crmmanage','rank','flatloan',
                                    'consumeloan','otherloan','foreigncurrency',
                                    'ctssign','creditstage','telebank',
                                    'messbank','netbank',
                                    'wechatbank','fastloan','gongjijin','daifa']
data_for_select = data_sample[candidates]

lg_m1 = forward_select(data=data_for_select, response='result')
lg_m1.summary().tables[1]
'''向前法选出来的变量
crmmanage + creditstage + marriage + foreigncurrency + messbank + 
consumeloan + netbank + ctssign + wechatbank + num + rank
'''
'''再次单变量验证显著性'''
list2=['marriage','creditstage','crmmanage','foreigncurrency','messbank',
       'consumeloan','netbank','ctssign','rank','wechatbank','num']
for i in list2:
    number=data_sample['{}'.format(i)].value_counts().count()
    cross_table = pd.crosstab(index=data_sample['{}'.format(i)],columns=data_sample.result)
    print(stats.chi2_contingency(cross_table.iloc[:number, :2]))
    print("------------------------------------")
'''选出来的分类变量可视化'''
import math
import seaborn as sns
from scipy import stats,integrate
import statsmodels.api as sm
plt.figure(figsize=(15,20), dpi=80)
n=1
for i in list2:
    plt.subplot(5,3,n)
    sns.barplot(x='{}'.format(i),y='result',data=data_sample)
    n+=1
'''num误差太大，建议删除，最后保留
crmmanage + creditstage + marriage + foreigncurrency + messbank + 
consumeloan + netbank + ctssign + wechatbank + rank
'''
#############连续变量筛选##################
list3=['avgaum', 'cash', 'cardmoney', 'loan', 'cts', 'allinvest', 'fundaum',
       'goldaum', 'bondaum', 'bankinvestaum', 'insaum', 'trustaum', 'otheraum',
       'alltrade', 'banktrade', 'autotrade', 'electrade', 'postrade',
       'othertrade', 'money_dbcrd', 'times_dbcrd', 'emoney_dbcrd',
       'etimes_dbcrd', 'money_crcrd', 'times_crcrd', 'emoney_crcrd',
       'etimes_crcrd', 'money_fp', 'times_fp', 'money_albb', 'times_albb',
       'money_tent', 'times_tent']
'''对极大值进行处理(不超过均值的3倍标准差)，盖帽法。特别注意：要对源数据进行处理'''
for i in list3:
    std=np.std(data_ori["{}".format(i)])
    mean=np.mean(data_ori["{}".format(i)])
    vmax=3*std+mean
    vmax_number=data_ori["{}".format(i)][data_ori["{}".format(i)]>vmax].count()
    print(i+":"+"{}".format(vmax_number))
    data_sample["{}".format(i)][data_sample["{}".format(i)]>vmax]=vmax
    data_target["{}".format(i)][data_target["{}".format(i)]>vmax]=vmax
'''相关系数矩阵'''
corrmatrix=data_sample.corr(method='pearson')
corrmatrix.result.sort_values(ascending=False)
'''查看多重共线性问题，选取相关系数0.65以上的'''
data_sample_lianxu=data_sample[['avgaum', 'cash', 'cardmoney', 'loan', 'cts', 'allinvest', 'fundaum',
       'goldaum', 'bondaum', 'bankinvestaum', 'insaum', 'trustaum', 'otheraum',
       'alltrade', 'banktrade', 'autotrade', 'electrade', 'postrade',
       'othertrade', 'money_dbcrd', 'times_dbcrd', 'emoney_dbcrd',
       'etimes_dbcrd', 'money_crcrd', 'times_crcrd', 'emoney_crcrd',
       'etimes_crcrd', 'money_fp', 'times_fp', 'money_albb', 'times_albb',
       'money_tent','times_tent']]
corrmatrix=data_sample_lianxu.corr(method='pearson')
corrmatrix[np.abs(corrmatrix)>0.65]
'''共线性较为严重，准备直接用PCA降维先
avgaum,allinvest,fundaum,insaum,alltrade,electrade,othertrade,times_crcrd,
剩'loan', 'cts','goldaum', 'bondaum','trustaum', 'otheraum','banktrade', 'postrade'等无需降维'''
data_sample_notpca=data_sample[['result','loan', 'cts','goldaum', 'bondaum','trustaum', 'otheraum','banktrade', 'postrade','cash','cardmoney','money_dbcrd','times_dbcrd','emoney_dbcrd','etimes_dbcrd',
                               'emoney_crcrd','etimes_crcrd','money_fp','times_fp','money_albb','times_albb','money_tent','times_tent','bankinvestaum','autotrade','money_crcrd']]
data_sample_pca = data_sample_lianxu.drop(['loan', 'cts','goldaum', 'bondaum','trustaum', 'otheraum','banktrade', 'postrade','cash','cardmoney','money_dbcrd','times_dbcrd','emoney_dbcrd','etimes_dbcrd',
                                          'emoney_crcrd','etimes_crcrd','money_fp','times_fp','money_albb','times_albb','money_tent','times_tent','bankinvestaum','autotrade','money_crcrd'],axis=1)
'''单变量方差分析'''
from statsmodels.formula.api import ols
for i in ['loan', 'cts','goldaum', 'bondaum','trustaum', 'otheraum','banktrade', 'postrade','cash','cardmoney','money_dbcrd','times_dbcrd','emoney_dbcrd','etimes_dbcrd',
                                          'emoney_crcrd','etimes_crcrd','money_fp','times_fp','money_albb','times_albb','money_tent','times_tent','bankinvestaum','autotrade','money_crcrd']:
    print(i+":")
    print(sm.stats.anova_lm(ols('result ~ {}'.format(i),data=data_sample_notpca).fit()))    #不显著的变量：postrade
corrmatrix_notpca=data_sample_notpca.corr(method='pearson')#spearman相关系数矩阵，可选pearson相关系数
corrmatrix_notpca[np.abs(corrmatrix_notpca)>0.65]
corrmatrix_notpca.result.sort_values(ascending=False)
data_sample_notpca.drop(['cash','etimes_dbcrd','times_dbcrd','money_albb','money_dbcrd'],axis=1,inplace=True)
corrmatrix_notpca=data_sample_notpca.corr(method='pearson')#spearman相关系数矩阵，可选pearson相关系数
corrmatrix_notpca[np.abs(corrmatrix_notpca)>0.65]
'''删除相关性低的共线性变量，有emoney_dbcrd,money_albb,money_tent,emoney_dbcrd,emoney_crcrd'''
data_sample_notpca.drop(['emoney_dbcrd','times_albb','emoney_crcrd','times_tent'],axis=1,inplace=True)
corrmatrix_notpca=data_sample_notpca.corr(method='pearson')#spearman相关系数矩阵，可选pearson相关系数
corrmatrix_notpca[np.abs(corrmatrix_notpca)>0.65]
corrmatrix_notpca.result.sort_values(ascending=False)
data_sample_notpca.columns
'''向前法'''
candidates = ['result', 'loan', 'cts', 'goldaum', 'bondaum', 'trustaum', 'otheraum',
       'banktrade', 'postrade', 'cardmoney', 'etimes_crcrd', 'money_fp',
       'times_fp', 'money_tent', 'bankinvestaum', 'autotrade', 'money_crcrd']
data_for_select = data_sample_notpca[candidates]
lg_m1 = forward_select(data=data_for_select, response='result')
lg_m1.summary().tables[1]
'''向前法选出变量
banktrade + autotrade + bankinvestaum + times_fp + money_crcrd + goldaum +
 money_tent + loan + bondaum + etimes_crcrd 
'''
data_sample_new=data_sample_notpca.drop('postrade',axis=1)
#data_sample_new=data_sample_notpca.drop(['postrade','cts','trustaum','otheraum','cardmoney','money_fp'],axis=1)
data_sample_new_scale=data_sample_new.drop("result",axis=1)
data_sample_new_scale.reset_index(drop=True,inplace=True)
############主成分分析##########
from sklearn.decomposition import PCA, FactorAnalysis,FastICA,SparsePCA
from sklearn import preprocessing
data_sample_pca.corr(method='pearson')
'''散点图矩阵'''
import seaborn as sns
sns.pairplot(data_sample_pca)
plt.show()
'''第一次做，先看下情况'''
data_sample_pca_scale = preprocessing.scale(data_sample_pca)    #数据中心标准化
propca=PCA(n_components=8,whiten=True)
newData=propca.fit(data_sample_pca_scale)
print(propca.explained_variance_)
print(propca.explained_variance_ratio_)
'''决定选取4个主成分'''
propca=PCA(n_components=4,whiten=True)
newData=propca.fit(data_sample_pca_scale)
print(propca.explained_variance_)
print(propca.explained_variance_ratio_)
pca_df=pd.DataFrame(propca.components_).T  #z主成分
'''需要再做因子转换'''
from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
fa = FactorAnalysis.load_data_samples(
        data_sample_pca_scale,
        preproc_demean=True,
        preproc_scale=True
        )
fa.extract_components()
fa.find_comps_to_retain(method='top_n',num_keep=4)
fa.rotate_components(method='varimax')
fa_plotting.graph_summary(fa)
fca_df=pd.DataFrame(fa.comps["rot"])       #因子转换后的主成分
'''将主成分含义展示
#0:'avgaum', 'allinvest','insaum'
#1:'alltrade', 'electrade'
#3:'othertrade', 'times_crcrd'
#4:fundaum'''
fa_score = pd.DataFrame(fa.get_component_scores(data_sample_pca_scale))
fa_score=fa_score.rename(columns={0: "assets", 1: "trades", 2: "crcrdtimes_othertrade",3:"funds"})
'''PCA后的散点图矩阵'''
import seaborn as sns
sns.pairplot(fa_score)
plt.show()
#############创建最终表##############
'''加入分类变量'''
data_sample_final=data_sample[['result','marriage','creditstage','crmmanage','foreigncurrency','messbank','consumeloan','netbank','ctssign','rank','wechatbank','age']]
'''加入PCA后的连续变量'''
data_sample_final=data_sample_final.join(fa_score)
'''加入非PCA的连续变量'''
data_sample_final=data_sample_final.join(data_sample_new_scale)
data_final = data_sample_final
corrmatrix_final=data_final.corr(method='pearson')
corrmatrix_final.result.sort_values(ascending=False)
corrmatrix_final[np.abs(corrmatrix_final)>0.65]
'''
['postrade','cts','trustaum','otheraum','cardmoney','money_fp']向前法中可以删除的
'''
'''查看分布'''
data_final[['age', 'loan', 'cts', 'goldaum', 'bondaum', 'trustaum', 'otheraum',
       'banktrade', 'cardmoney', 'money_fp', 'times_fp', 'money_tent',
       'bankinvestaum', 'autotrade', 'money_crcrd']].hist(figsize=(40,30),bins=30)
'''正态性转换'''
data_zhengtai=np.log(data_final[['age', 'loan', 'cts', 'goldaum', 'bondaum', 'trustaum', 'otheraum',
       'banktrade', 'cardmoney', 'money_fp', 'times_fp', 'money_tent',
       'bankinvestaum', 'autotrade', 'money_crcrd']]+0.1)
data_zhengtai['result']=data_final['result']
data_zhengtai[['age', 'loan', 'cts', 'goldaum', 'bondaum', 'trustaum', 'otheraum',
       'banktrade', 'cardmoney', 'money_fp', 'times_fp', 'money_tent',
       'bankinvestaum', 'autotrade', 'money_crcrd']].hist(figsize=(40,30),bins=30)
'''正态分布的相关性系数矩阵'''
corrmatrix_zhengtai=data_zhengtai.corr(method='pearson')
corrmatrix_zhengtai.result.sort_values(ascending=False) 
'''原矩阵'''
corrmatrix_final.result.sort_values(ascending=False) 
'''将正态性转换后的变量替换进去'''
data_final_last=data_final.drop(['bankinvestaum','goldaum','money_crcrd','cardmoney','otheraum',
                                 'times_fp','money_fp','loan','cts','money_tent'],axis=1)
data_final_last[['lnbankinvestaum','lngoldaum','lnmoney_crcrd','lncardmoney','lnotheraum',
                 'lntimes_fp','lnmoney_fp','lnloan','lncts','lnmoney_tent']]=data_zhengtai[['bankinvestaum',
                'goldaum','money_crcrd','cardmoney','otheraum','times_fp','money_fp','loan','cts','money_tent']]
corrmatrix_last=data_final_last.corr(method='pearson')
corrmatrix_last.result.sort_values(ascending=False) 
'''连续变量向前法剔除的变量还剩trustaum，也删掉'''
data_final_last.drop('trustaum',axis=1,inplace=True)
corrmatrix_last[np.abs(corrmatrix_last)>0.65]       #查看共线性问题
'''去除lncts,lnmoney_fp.lnmoney_tent,etimes_crcrd,lnmoney_crcrd'''
data_final_last.drop(['lnmoney_fp','lnmoney_tent','lncts','etimes_crcrd','lnmoney_crcrd'],axis=1,inplace=True)
corrmatrix_last=data_final_last.corr(method='pearson')
corrmatrix_last[np.abs(corrmatrix_last)>0.65]
corrmatrix_last.result.sort_values(ascending=False) 
'''多分类变量的处理'''
bins = [0,20,30,40,50,60,70]
data_final_last['age_bins'] = pd.cut(data_final_last['age'],bins,labels=False)
data_final_last=data_final_last.join(pd.get_dummies(data_final_last['marriage'],prefix='marriage'))
data_final_last=data_final_last.join(pd.get_dummies(data_final_last['rank'],prefix='rank'))
data_final_last=data_final_last.join(pd.get_dummies(data_final_last['age_bins'],prefix='age'))
data_final_last.drop(['marriage','rank','age','age_bins'],axis=1,inplace=True)
data_final_last.isnull().sum()    #有缺失值，用中位数填补
data_final_last.lnotheraum.fillna(data_final_last.lnotheraum.median(),inplace=True)
datax=data_final_last.drop('result',axis=1)
datay=pd.DataFrame(data_final_last['result'])
###########建模###############
'''切分训练集和测试集'''
import sklearn.cross_validation as cross_validation
train_data, test_data, train_target, test_target = cross_validation.train_test_split(datax, datay, test_size=0.4, random_state=1234)
'''决策树'''
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5)
clf_fit = clf.fit(train_data, train_target)
test_est_tr = clf.predict(test_data)
train_est_tr = clf.predict(train_data)
test_est_p_tr = clf.predict_proba(test_data)[:,1]
train_est_p_tr = clf.predict_proba(train_data)[:,1]
import sklearn.metrics as metrics
print (metrics.classification_report(test_target, test_est_tr))
print (metrics.classification_report(train_target, train_est_tr))
fpr_test_tr, tpr_test_tr, th_test_tr = metrics.roc_curve(test_target, test_est_p_tr)
'''组合算法'''
import sklearn.cross_validation as cross_validation
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.feature_selection as feature_selection
import sklearn.metrics as metrics
from sklearn.feature_selection import RFE
from sklearn.externals.six import StringIO
for i in range(10,301,10):
    abc = ensemble. AdaBoostClassifier(n_estimators=i)
    abc.fit(train_data, train_target)
    test_est_abc = abc.predict(test_data)
    test_est_p_abc = abc.predict_proba(test_data)[:,1]
    fpr_test_abc, tpr_test_abc, th_test_abc = metrics.roc_curve(test_target, test_est_p_abc)
    print (metrics.classification_report(test_target, test_est_abc))
abc = ensemble. AdaBoostClassifier(n_estimators=80)
abc.fit(train_data, train_target)
test_est_abc = abc.predict(test_data)
test_est_p_abc = abc.predict_proba(test_data)[:,1]
fpr_test_abc, tpr_test_abc, th_test_abc = metrics.roc_curve(test_target, test_est_p_abc)
print (metrics.classification_report(test_target, test_est_abc))
rfc = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=3, max_features=0.5, min_samples_split=5)
rfc.fit(train_data, train_target)
test_est_rfc = rfc.predict(test_data)
test_est_p_rfc = rfc.predict_proba(test_data)[:,1]
fpr_test_rfc, tpr_test_rfc, th_test_rfc = metrics.roc_curve(test_target, test_est_p_rfc)
print (metrics.classification_report(test_target, test_est_rfc))
'''逻辑回归'''
logistic_model = linear_model.LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l1', random_state=None, tol=0.001)
logistic_model.fit(train_data, train_target)
test_est_lg = logistic_model.predict(test_data)
train_est_lg = logistic_model.predict(train_data)
test_est_p_lg = logistic_model.predict_proba(test_data)[:,1]
train_est_p_lg = logistic_model.predict_proba(train_data)[:,1]
fpr_test_lg, tpr_test_lg, th_test_lg = metrics.roc_curve(test_target, test_est_p_lg)
print (metrics.classification_report(test_target, test_est_lg))
'''ROC'''
plt.plot(fpr_test_tr, tpr_test_tr, "b-")
plt.plot(fpr_test_lg, tpr_test_lg, "r--")
#plt.plot(fpr_test_svc, tpr_test_svc, color="yellow")
plt.plot(fpr_test_abc, tpr_test_abc, "y-")
plt.plot(fpr_test_rfc, tpr_test_rfc, "g--")
plt.title('ROC curve')
print('decision tree accuracy: AUC = %.4f' %metrics.auc(fpr_test_tr, tpr_test_tr))
print('logistic regression accuracy: AUC = %.4f' %metrics.auc(fpr_test_lg, tpr_test_lg))
print('abc classifier accuracy: AUC = %.4f' %metrics.auc(fpr_test_abc, tpr_test_abc))
print('random forest accuracy: AUC = %.4f' %metrics.auc(fpr_test_rfc, tpr_test_rfc))
import seaborn as sns
test_target=np.squeeze(test_target)
red, blue = sns.color_palette("Set1",2)
sns.kdeplot(test_est_p_abc[test_target==1], shade=True, color=red)
sns.kdeplot(test_est_p_abc[test_target==0], shade=True, color=blue)
'''交叉验证'''
train_data=np.squeeze(train_data)
train_target=np.squeeze(train_target)
lr = linear_model.LogisticRegression()
lr_scores = cross_validation.cross_val_score(lr, train_data, train_target, cv=5)
print("logistic regression accuracy:")
print(lr_scores)

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5)
clf_scores = cross_validation.cross_val_score(clf, train_data, train_target, cv=5)
print("decision tree accuracy:")
print(clf_scores)

abc_scores = cross_validation.cross_val_score(abc, train_data, train_target, cv=5)
print("abc classifier accuracy:")
print(abc_scores)

rfc_scores = cross_validation.cross_val_score(rfc, train_data, train_target, cv=5)
print("random forest accuracy:")
print(rfc_scores)
##################保存模型#####################
import pickle as pickle
model_file = open(r'abc0902.model', 'wb')
pickle.dump(abc, model_file)
model_file.close()
'''用已有数据进行预测'''

'''读取模型'''
model_load_file = open(r'abc0902.model', 'rb')
model_load = pickle.load(model_load_file)
model_load_file.close()
print (pd.DataFrame(list(zip(datax.columns, model_load.feature_importances_)), columns=['feature', 'importance']))
##################准备新的预测数据#####################
'''准备分类变量'''
data_target_new=data_target[['creditstage', 'crmmanage', 'foreigncurrency', 'messbank',
       'consumeloan', 'netbank', 'ctssign', 'wechatbank','marriage','rank','age']]
data_target_new.reset_index(drop=True,inplace=True)
'''准备PCA连续变量'''
data_target_pca=data_target[['avgaum','allinvest','fundaum','insaum','alltrade','electrade','othertrade','times_crcrd']]
data_sample_pca_scale = preprocessing.scale(data_target_pca) 
fa_score = pd.DataFrame(fa.get_component_scores(data_sample_pca_scale))
fa_score=fa_score.rename(columns={0: "assets", 1: "trades", 2: "crcrdtimes_othertrade",3:"funds"})
data_target_ln=data_target[['bondaum', 'trustaum',
       'banktrade', 'autotrade', 'bankinvestaum', 'goldaum',
       'money_crcrd', 'cardmoney', 'otheraum', 'times_fp', 'loan',
       'cts']]

data_zhengtai=np.log(data_target_ln[['bondaum', 'trustaum',
       'banktrade', 'autotrade', 'bankinvestaum', 'goldaum',
       'money_crcrd', 'cardmoney', 'otheraum', 'times_fp', 'loan',
       'cts']]+0.1)

data_target_ln=data_target_ln.drop([ 'bankinvestaum', 'goldaum',
       'money_crcrd', 'cardmoney', 'otheraum', 'times_fp', 'loan',
       'cts'],axis=1)

data_target_ln[['lnbankinvestaum', 'lngoldaum',
       'lnmoney_crcrd', 'lncardmoney', 'lnotheraum', 'lntimes_fp', 'lnloan',
       'lncts']]=data_zhengtai[[ 'bankinvestaum', 'goldaum',
       'money_crcrd', 'cardmoney', 'otheraum', 'times_fp', 'loan',
       'cts']]
data_target_ln.reset_index(drop=True,inplace=True)
data_target_final=data_target_new.join(fa_score)
data_target_final=data_target_final.join(data_target_ln)
bins = [0,20,30,40,50,60,70]
data_target_final['age_bins'] = pd.cut(data_target_final['age'],bins,labels=False)
data_target_final=data_target_final.join(pd.get_dummies(data_target_final['marriage'],prefix='marriage'))
data_target_final=data_target_final.join(pd.get_dummies(data_target_final['rank'],prefix='rank'))
data_target_final=data_target_final.join(pd.get_dummies(data_target_final['age_bins'],prefix='age'))
data_target_final.drop(['marriage','rank','age','age_bins'],axis=1,inplace=True)
data_target_final.isnull().sum()
data_target_final.lnotheraum.fillna(data_target_final.lnotheraum.median(),inplace=True)
data_target_final.drop(['trustaum','lnmoney_crcrd','lncts'],axis=1,inplace=True)
'''发现marriage中少了1个类型，需自行补上'''
data_target_final['marriage_22']=data_target_final['marriage_20']
data_target_final['marriage_22']=data_target_final['marriage_22'].replace(1,0)
data_target_final_new=pd.DataFrame()
data_target_final_new[['creditstage', 'crmmanage', 'foreigncurrency', 'messbank',
       'consumeloan', 'netbank', 'ctssign', 'wechatbank', 'assets', 'trades',
       'crcrdtimes_othertrade', 'funds', 'bondaum', 'banktrade', 'autotrade',
       'lnbankinvestaum', 'lngoldaum', 'lncardmoney', 'lnotheraum',
       'lntimes_fp', 'lnloan', 'marriage_000', 'marriage_10', 'marriage_20',
       'marriage_22', 'marriage_30', 'marriage_40', 'marriage_90',
       'marriage_99', 'rank_801.0', 'rank_802.0', 'rank_803.0', 'rank_999.0',
       'rank_1802.0', 'rank_1902.0', 'rank_1903.0', 'rank_1904.0', 'age_0',
       'age_1', 'age_2', 'age_3', 'age_4', 'age_5']]=data_target_final[['creditstage', 'crmmanage', 'foreigncurrency', 'messbank',
       'consumeloan', 'netbank', 'ctssign', 'wechatbank', 'assets', 'trades',
       'crcrdtimes_othertrade', 'funds', 'bondaum', 'banktrade', 'autotrade',
       'lnbankinvestaum', 'lngoldaum', 'lncardmoney', 'lnotheraum',
       'lntimes_fp', 'lnloan', 'marriage_000', 'marriage_10', 'marriage_20',
       'marriage_22', 'marriage_30', 'marriage_40', 'marriage_90',
       'marriage_99', 'rank_801.0', 'rank_802.0', 'rank_803.0', 'rank_999.0',
       'rank_1802.0', 'rank_1902.0', 'rank_1903.0', 'rank_1904.0', 'age_0',
       'age_1', 'age_2', 'age_3', 'age_4', 'age_5']]
test_est_load=model_load.predict(data_target_final_new)
test_est_p_load=model_load.predict_proba(data_target_final_new)[:,1]
test_est_df_p = pd.DataFrame(test_est_p_load,columns=['targetpro'])
test_est_df = pd.DataFrame(test_est_load,columns=['target'])
final=test_est_df.join(data_target_final_new)
final=final.join(test_est_df_p)
print(test_est_df.target.value_counts())
data_target.reset_index(drop=True,inplace=True)
final=final.join(data_target.newid)
final.to_excel("6~7月INS数据挖掘.xlsx")

