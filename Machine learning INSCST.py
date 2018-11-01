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
import sklearn.feature_selection as feature_selection
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import operator

'''设置行列显示数'''
pd.set_option('max_columns',100)
pd.set_option('max_rows',100)
'''导入数据'''
data_ori=pd.read_excel("INS_CST.xlsx")    #读取数据
data_ori['data']=data_ori['data'].astype(pd.datetime)    #转换成日期格式
data_ori=data_ori.sort_values(by='data')
#print(data_ori.head(10))

##################一、基本统计分析######################
'''1-1划分建模用数据和预测数据'''
data_JantoMay = data_ori[data_ori.result.notnull()]
data_JuJy = data_ori[data_ori.result.isnull()]
xn,yn=data_JantoMay.shape    #行和列数
cols = data_JantoMay.columns.tolist()       #变量列表
#scan = data_JantoMay.describe(include='all').T      #总览
'''1-2查看整体样本响应度'''
allResponse = data_JantoMay.result.value_counts()[1]/data_JantoMay.result.value_counts().sum()
'''1-2-1查看各个分类变量的响应度和数量'''
for i in cols[6:28]:
    cross_tab = pd.crosstab(index=data_JantoMay["{}".format(i)], columns=data_JantoMay.result)
    lost = data_JantoMay["{}".format(i)].isnull().sum()
    print("{}类别数量".format(i))
    print(cross_tab.sum(1))
    print("缺失量：{}，缺失率：{}".format(lost,lost/len(data_JantoMay["{}".format(i)])))
    print("{}类别响应度".format(i))
    print(cross_tab.div(cross_tab.sum(1),axis=0))
    print("========================")
'''结果：
===================正常区======================
相对正常的:'crmmanage','ctssign','messbank','netbank','daifa'
===================观察区======================
分类不均匀:'marriage','rank'
分类集中度过高:'flatloan','consumeloan','otherloan','foreigncurrency','creditstage',
              'mobilebank','wechatbank','fastloan'
高缺失变量:'edu','familynum'
响应度差异小:'gender','telebank','gongjijin'
===================删除区======================
无区分,直接删除:'businessloan','carloan'
'''
data_JantoMay.drop(['businessloan','carloan'],axis=1,inplace=True)
cols.remove('businessloan');cols.remove('carloan')
'''1-2-2查看各个连续变量的数量和缺失率'''
for i in cols[26:]:
    valueCount = np.array(data_JantoMay["{}".format(i)].value_counts())
    mostValue = data_JantoMay["{}".format(i)].value_counts().index[0]
    mostPre = valueCount[0]/len(data_JantoMay["{}".format(i)])
    lost = data_JantoMay["{}".format(i)].isnull().sum()
    print("{}类最高值占比".format(i))
    print("{}:{}".format(mostValue,mostPre))
    print("缺失量：{}，缺失率：{}".format(lost,lost/len(data_JantoMay["{}".format(i)])))
    print("========================")
'''结果：
===================正常区======================
相对正常的:'avgaum','cash','cardmoney','allinvest','fundaum','bankinvestaum','insaum',
          'alltrade','banktrade','autotrade','electrade','monet_dbcrd','times_dbcrd',
          'emoney_dbcrd','etimes_dbcrd','money_fp','times_fp','money_albb','times_albb','age'
===================观察区======================
分类集中度过高:'loan','cts','goldaum','bondaum','trustaum','otheraum','postrade','money_crcrd',
              'times_crcrd','emoney_crcrd','etimes_crcrd','money_tent','times_tent'
===================删除区======================
暂无
'''
'''1-3填补缺失值'''
data_JantoMay['age']=data_JantoMay['age'].fillna(np.mean(data_JantoMay.age))    #用中位数填补
data_JantoMay.marriage=data_JantoMay.marriage.fillna('@')    #填补婚姻缺失值,缺失较多，且响应差异较大，故自成一类
data_JantoMay.marriage.replace(to_replace='@',value='@000',inplace=True)     #替换婚姻状态
data_JantoMay['rank']=data_JantoMay['rank'].fillna(1802)    #填补等级缺失值，缺失较多，且响应差异较大，故自成一类
data_JantoMay['rank'][data_JantoMay['rank']==1904] = 1902
data_JantoMay.edu=data_JantoMay.edu.fillna('@')    
data_JantoMay.gender = data_JantoMay.gender.fillna(99)
data_JantoMay[['familynum','telebank','gongjijin','daifa']]=data_JantoMay[['familynum','telebank','gongjijin','daifa']].fillna(0)
data_JantoMay[[ 'alltrade', 'banktrade', 'autotrade', 'electrade', 'postrade',
       'othertrade', 'money_dbcrd', 'times_dbcrd', 'emoney_dbcrd',
       'etimes_dbcrd', 'money_crcrd', 'times_crcrd', 'emoney_crcrd',
       'etimes_crcrd', 'money_fp', 'times_fp', 'money_albb', 'times_albb',
       'money_tent', 'times_tent']]=data_JantoMay[[ 'alltrade', 'banktrade', 'autotrade', 'electrade', 'postrade',
       'othertrade', 'money_dbcrd', 'times_dbcrd', 'emoney_dbcrd',
       'etimes_dbcrd', 'money_crcrd', 'times_crcrd', 'emoney_crcrd',
       'etimes_crcrd', 'money_fp', 'times_fp', 'money_albb', 'times_albb',
       'money_tent', 'times_tent']].fillna(0)    #交易信息、消费信息、第三方信息,未取到，就是0
data_JantoMay[['flatloan',
       'consumeloan', 'otherloan', 'foreigncurrency', 'ctssign', 'creditstage',
        'messbank', 'netbank', 'mobilebank', 'wechatbank',
       'fastloan',   'avgaum', 'cash',
       'cardmoney', 'loan', 'cts', 'allinvest', 'fundaum', 'goldaum',
       'bondaum', 'bankinvestaum', 'insaum', 'trustaum', 'otheraum',
]]=data_JantoMay[['flatloan', 
       'consumeloan', 'otherloan', 'foreigncurrency', 'ctssign', 'creditstage',
        'messbank', 'netbank', 'mobilebank', 'wechatbank',
       'fastloan',  'avgaum', 'cash',
       'cardmoney', 'loan', 'cts', 'allinvest', 'fundaum', 'goldaum',
       'bondaum', 'bankinvestaum', 'insaum', 'trustaum', 'otheraum',
]].fillna(0)    #产品覆盖信息、aum信息，未取到，就是0
scan = data_JantoMay.describe(include='all').T
'''1-4分类变量特征工程'''
'''1-4-1删除不用的变量，newid,num,sumins,data'''
data_JantoMay.reset_index(drop=True,inplace=True)     #重设index
data_JantoMay.drop(['newid','num','sumins','data'],axis=1,inplace=True)
cols.remove('newid');cols.remove('num');cols.remove('sumins');cols.remove('data')
'''1-4-2相关系数初步筛选'''
corrMa = data_JantoMay.corr()
print(corrMa.result.sort_values(ascending=False))
'''1-4-3分类变量配上卡方检验'''
ChiMa = np.array(feature_selection.chi2(data_JantoMay.iloc[:,4:22],data_JantoMay['result'])).T
for i in ChiMa[:,0].argsort( ):
    print(cols[4:22][i],':',ChiMa[i,:])
'''还有marriage和edu，不是数字'''
print(stats.chi2_contingency(pd.crosstab(index=data_JantoMay.edu,columns=data_JantoMay.result)))
print(stats.chi2_contingency(pd.crosstab(index=data_JantoMay.marriage,columns=data_JantoMay.result)))

'''结合相关系数和卡方检验的初步判断
======删除的变量=======
'mobilebank','gender','gongjijin','telebank','otherloan','flatloan'
=========犹豫的===========
'fastloan','wechatbank','familynum','foreigncurrency','consumeloan','creditstage','edu','marriage'
=========暂时保留=========
'netbank','daifa','messbank','ctssign','crmmanage','rank'
'''
'''1-4-4分类变量IV值'''
def CalIV(data,label):
    eps = 1e-6
    tab = pd.crosstab(data,label)
    all0 = np.sum(label==0)
    all1 = np.sum(label==1)
    IV = 0
    for i in range(len(tab)):
        group0 = tab.iloc[i,0]
        group1 = tab.iloc[i,1]
        IV += ((group0/all0+eps) - (group1/all1+eps))*np.log((group0/all0+eps)/((group1/all1+eps)+eps))
    return IV
'''先看初步决定删除的变量'''
for i in ['mobilebank','gender','gongjijin','telebank','otherloan','flatloan']:
    print(i,CalIV(data_JantoMay['{}'.format(i)],data_JantoMay.result))
'''
mobilebank 0.0300530107449
gender 0.0455448097632
gongjijin 0.0190088603494
telebank 0.0423407937495
otherloan 0.00968813954269
flatloan 0.0103361378883
'''
for i in ['fastloan','wechatbank','familynum','foreigncurrency','consumeloan','creditstage','edu','marriage']:
    print(i,CalIV(data_JantoMay['{}'.format(i)],data_JantoMay.result))
'''
fastloan 0.0171332488725
wechatbank 0.0242484908956
familynum 0.033386063157
foreigncurrency 0.185914748078
consumeloan 0.14321306399
creditstage 0.190424659443
edu 0.357467769489
marriage 0.282008402191
'''
for i in ['netbank','daifa','messbank','ctssign','crmmanage','rank']:
    print(i,CalIV(data_JantoMay['{}'.format(i)],data_JantoMay.result))
'''
netbank 0.0897031569798
daifa 0.0581655204646
messbank 0.22095603023
ctssign 0.112686996894
crmmanage 0.278554974876
rank 0.41184937843
'''
'''结论：
第一部分不显著的全部删掉
第二部分'fastloan','wechatbank','familynum'删除
第三部分'netbank','daifa'进入观察区
剩余从高到底依次是'rank','edu','marriage','crmmanage','messbank','creditstage',
'foreigncurrency','consumeloan','ctssign'
'''
data_feature = data_JantoMay.drop(['mobilebank','gender','gongjijin','telebank',
                                   'otherloan','flatloan','fastloan','wechatbank','familynum'],axis=1)
'''1-4-5向前法'''
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
candidates = ['result','rank','edu','marriage','crmmanage','messbank','creditstage',
'foreigncurrency','consumeloan','ctssign','netbank','daifa']
data_for_select = data_JantoMay[candidates]
lg_m1 = forward_select(data=data_for_select, response='result')
lg_m1.summary().tables[1]
'''向前法选出来的变量
edu + crmmanage + creditstage + messbank + foreigncurrency + consumeloan + 
netbank + ctssign + daifa '''
'''1-4-6最终分类变量暂时保留'rank','edu','marriage','crmmanage','messbank','creditstage',
'foreigncurrency','consumeloan','ctssign','netbank','daifa''''
#对marriage和edu的处理
data_feature['marriage'][data_feature['marriage']=='@40'] = '@20'
data_feature['marriage'][data_feature['marriage']=='@30'] = '@20'
data_feature['marriage'][data_feature['marriage']=='@22'] = '@20'
print(CalIV(data_feature.marriage,data_feature.result))     #0.280527483169
temp = data_feature.edu.value_counts().index[12:].tolist()
for i in temp:
    data_feature['edu'][data_feature['edu']=='{}'.format(i)] = '@'
print(CalIV(data_feature.edu,data_feature.result))     #0.315429798549
'''1-4-7分类变量响应度和误差的可视化'''
CateList = ['rank','edu','marriage','crmmanage','messbank','creditstage',
'foreigncurrency','consumeloan','ctssign','netbank','daifa']
import math
import seaborn as sns
from scipy import stats,integrate
import statsmodels.api as sm
plt.figure(figsize=(15,20), dpi=80)
n=1
for i in list2:
    plt.subplot(5,3,n)
    sns.barplot(x='{}'.format(i),y='result',data=data_feature)
    n+=1
'''1-5连续变量特征工程选取'''
'''1-5-1初步处理'''
ConList=['avgaum', 'cash', 'cardmoney', 'loan', 'cts', 'allinvest', 'fundaum',
       'goldaum', 'bondaum', 'bankinvestaum', 'insaum', 'trustaum', 'otheraum',
       'alltrade', 'banktrade', 'autotrade', 'electrade', 'postrade',
       'othertrade', 'money_dbcrd', 'times_dbcrd', 'emoney_dbcrd',
       'etimes_dbcrd', 'money_crcrd', 'times_crcrd', 'emoney_crcrd',
       'etimes_crcrd', 'money_fp', 'times_fp', 'money_albb', 'times_albb',
       'money_tent', 'times_tent']
'''对极大值进行处理(不超过均值的5倍标准差)，盖帽法。特别注意：要对源数据进行处理'''
for i in ConList:
    std=np.std(data_JantoMay["{}".format(i)])
    mean=np.mean(data_JantoMay["{}".format(i)])
    vmax=5*std+mean
    vmax_number=data_JantoMay["{}".format(i)][data_JantoMay["{}".format(i)]>vmax].count()
    print(i+":"+"{}".format(vmax_number))
    data_JantoMay["{}".format(i)][data_JantoMay["{}".format(i)]>vmax]=vmax
    data_JantoMay["{}".format(i)][data_JantoMay["{}".format(i)]>vmax]=vmax
'''1-5-2根据相关系数选取变量，并查看多重共线性问题，选取相关系数0.6以上的'''
data_lianxu=data_JantoMay[['result','age','avgaum', 'cash', 'cardmoney', 'loan', 'cts', 'allinvest', 'fundaum',
       'goldaum', 'bondaum', 'bankinvestaum', 'insaum', 'trustaum', 'otheraum',
       'alltrade', 'banktrade', 'autotrade', 'electrade', 'postrade',
       'othertrade', 'money_dbcrd', 'times_dbcrd', 'emoney_dbcrd',
       'etimes_dbcrd', 'money_crcrd', 'times_crcrd', 'emoney_crcrd',
       'etimes_crcrd', 'money_fp', 'times_fp', 'money_albb', 'times_albb',
       'money_tent','times_tent']]
conMa=data_lianxu.corr(method='pearson')
print(conMa.result.sort_values(ascending=False))
''''age','loan', 'cts','goldaum', 'bondaum','trustaum', 'otheraum','banktrade', 'postrade'共线性低'''
'''1-5-3单变量方差分析'''
from statsmodels.formula.api import ols
for i in ['result','age','loan', 'cts','goldaum', 'bondaum','trustaum', 'otheraum','banktrade', 'postrade']:
    print(i+":")
    print(sm.stats.anova_lm(ols('result ~ {}'.format(i),data=data_feature).fit()))    
'''
=========需删除的=========
'cts','trustaum','otheraum','postrade'
=========观察=============
'loan','goldaum','bondaum'
=========保留=============
'banktrade'
'''
data_feature.drop(['cts','trustaum','otheraum','postrade'],axis=1,inplace=True)
'''1-5-4共线性较为严重，进行处理'''
data_pca = data_lianxu.drop(['loan', 'cts','goldaum', 'bondaum','trustaum', 'otheraum','banktrade', 'postrade'],axis=1)
pcaMa = data_pca.corr()
print(pcaMa.result.sort_values(ascending=False))
'''单变量方差分析'''
for i in data_pca.columns.tolist():
    print(i+":")
    print(sm.stats.anova_lm(ols('result ~ {}'.format(i),data=data_pca).fit()))    
#都是挺显著的变量，因此要去除共线性
'''先尝试直接删除：
资产类：'avgaum','insaum','allinvest','cash'
渠道类：'alltrade'
其他交易类：'times_fp','times_albb','times_crcrd','etimes_dbcrd','emoney_crcrd','times_tent',
'emoney_dbcrd','money_albb','money_tent','etimes_crcrd'
'''
data_pca.drop(['avgaum','insaum','alltrade','times_fp','times_albb','times_crcrd',
               'etimes_dbcrd','emoney_crcrd','times_tent','emoney_dbcrd','money_albb',
               'money_tent','allinvest','cash','etimes_crcrd'],axis=1,inplace=True)
'''1-5-5得到连续变量集'''
data_lianxu.drop(['cts','trustaum','otheraum','postrade','avgaum','insaum','alltrade','times_fp','times_albb','times_crcrd',
               'etimes_dbcrd','emoney_crcrd','times_tent','emoney_dbcrd','money_albb',
               'money_tent','allinvest','cash','etimes_crcrd'],axis=1,inplace=True)
conMa = data_lianxu.corr()
print(conMa.result.sort_values(ascending=False))
'''1-5-6向前法选出变量'''
lg_m1 = forward_select(data=data_lianxu, response='result')
lg_m1.summary().tables[1]
'''
fundaum + othertrade + banktrade + autotrade + cardmoney + electrade + 
bankinvestaum + times_dbcrd + bondaum + loan + money_fp,
删除了'money_crcrd','goldaum','money_dbcrd','age' 
'''
data_lianxu.drop(['money_crcrd','goldaum','money_dbcrd','age'],axis=1,inplace=True)    
'''1-5-7连续变量标准化，并创建最终表'''  
ConList = data_lianxu.columns.tolist()
from sklearn import preprocessing
data_lianxu_scale=preprocessing.scale(data_lianxu.drop(['result'],axis=1))
final_data = data_feature[ConList]
final_data[CateList] = data_feature[CateList]
ConList.remove('result')
n = 0
for i in ConList:
    final_data['{}'.format(i)] = data_lianxu_scale[:,n]
    n+=1
final_data=final_data.join(pd.get_dummies(final_data['rank'],prefix='rank'))
final_data=final_data.join(pd.get_dummies(final_data['marriage'],prefix='marriage'))
final_data=final_data.join(pd.get_dummies(final_data['edu'],prefix='edu'))
final_data.drop(['rank','marriage','edu'],axis=1,inplace=True)   

'''1-6建模''' 
final_data.isnull().sum()    #有缺失值，用中位数填补
#data_final_last.lnotheraum.fillna(data_final_last.lnotheraum.median(),inplace=True)
datax=final_data.drop('result',axis=1)
datay=pd.DataFrame(final_data['result'])
import sklearn.cross_validation as cross_validation
train_data, test_data, train_target, test_target = cross_validation.train_test_split(datax, datay, test_size=0.4, random_state=1234)
'''决策树'''
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_split=5)
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
'''1-7保存和读取模型'''
import pickle as pickle
model_file = open(r'abc1102.model', 'wb')
pickle.dump(abc, model_file)
model_file.close()
'''用已有数据进行预测'''

'''读取模型'''
model_load_file = open(r'abc0902.model', 'rb')
model_load = pickle.load(model_load_file)
model_load_file.close()
print (pd.DataFrame(list(zip(datax.columns, model_load.feature_importances_)), columns=['feature', 'importance']))

re=abc.predict_proba(datax)













    
    
    
    
    
    
    
    
    
corrmatrix_notpca=data_notpca.corr(method='pearson')#spearman相关系数矩阵，可选pearson相关系数
corrmatrix_notpca[np.abs(corrmatrix_notpca)>0.65]
corrmatrix_notpca.result.sort_values(ascending=False)
'''向前法'''
candidates = ['result', 'loan', 'cts','goldaum', 'bondaum','trustaum', 'otheraum','banktrade', 'postrade','money_dbcrd',
                               'money_albb','money_tent','bankinvestaum','autotrade','money_crcrd']
data_for_select = data_notpca[candidates]
lg_m1 = forward_select(data=data_for_select, response='result')
lg_m1.summary().tables[1]
'''向前法选出变量
banktrade + autotrade + bankinvestaum + money_crcrd + goldaum + money_albb + loan + bondaum'''
data_sample_new=data_notpca[['banktrade' ,'autotrade','bankinvestaum','money_crcrd','goldaum', 'money_albb','loan','bondaum']]
from sklearn.decomposition import PCA, FactorAnalysis,FastICA,SparsePCA
from sklearn import preprocessing
data_sample_new=preprocessing.scale(data_sample_new)
data_sample_new=pd.DataFrame(data_sample_new)
data_sample_new=data_sample_new.rename(columns={0: "banktrade", 1: "autotrade", 2: "bankinvestaum",3:"money_crcrd",4:"goldaum",5:"money_albb",6:"loan",7:"bondaum"})
#data_sample_new['result'] = data_notpca['result']
############主成分分析##########
corrmatrix_pca = data_pca.corr(method='pearson')
'''散点图矩阵'''
import seaborn as sns
sns.pairplot(data_pca)
plt.show()
'''第一次做，先看下情况'''
data_pca_scale = preprocessing.scale(data_pca)    #数据中心标准化
propca=PCA(n_components=17,whiten=True)
newData=propca.fit(data_pca_scale)
print(propca.explained_variance_)
print(propca.explained_variance_ratio_)
'''决定选取5个主成分'''
propca=PCA(n_components=5,whiten=True)
newData=propca.fit(data_pca_scale)
print(propca.explained_variance_)
print(propca.explained_variance_ratio_)
pca_df=pd.DataFrame(propca.components_).T  #z主成分
'''需要再做因子转换'''
from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
fa = FactorAnalysis.load_data_samples(
        data_pca_scale,
        preproc_demean=True,
        preproc_scale=True
        )
fa.extract_components()
fa.find_comps_to_retain(method='top_n',num_keep=5)
fa.rotate_components(method='varimax')
fa_plotting.graph_summary(fa)
fca_df=pd.DataFrame(fa.comps["rot"])       #因子转换后的主成分
'''将主成分含义展示
#0: 'times_dbcrd', 'etimes_dbcrd','times_fp','times_albb', 'times_tent'
#1:'avgaum', 'allinvest', 'fundaum', 'insaum'
#2: 'times_crcrd', 'emoney_crcrd', 'etimes_crcrd'
#3:'emoney_dbcrd','money_fp'
#4:'alltrade', 'electrade','othertrade'
'''
fa_score = pd.DataFrame(fa.get_component_scores(data_pca_scale))
fa_score=fa_score.rename(columns={0: "fp_times", 1: "assets", 2: "crcrd",3:"moneyfp",4:"trades"})
'''PCA后的散点图矩阵'''
import seaborn as sns
sns.pairplot(fa_score)
plt.show()

#############创建最终表##############
'''加入分类变量'''
data_sample_final=data_JantoMay[['result','crmmanage', 'creditstage' , 'marriage' ,'foreigncurrency' , 'messbank' ,'gender' ,
         'mobilebank','consumeloan' , 'netbank' , 'ctssign' , 'wechatbank']]
'''加入PCA后的连续变量'''
data_sample_final=data_sample_final.join(fa_score)
'''加入非PCA的连续变量'''
data_sample_final=data_sample_final.join(data_sample_new)
data_final = data_sample_final
corrmatrix_final=data_final.corr(method='pearson')
#删除money_crcrd,money_albb
data_final.drop(['money_crcrd','money_albb'],axis=1,inplace=True)
corrmatrix_final.result.sort_values(ascending=False)
'''查看分布'''
data_final[['fp_times', 'assets', 'crcrd', 'moneyfp', 'trades',
       'banktrade', 'autotrade', 'bankinvestaum', 'goldaum', 'loan',
       'bondaum']].hist(figsize=(40,30),bins=30)
'''正态性转换'''
data_zhengtai=np.log(data_final[['fp_times', 'assets', 'crcrd', 'moneyfp', 'trades',
       'banktrade', 'autotrade', 'bankinvestaum', 'goldaum', 'loan',
       'bondaum']]+1.0e6)
data_zhengtai['result']=data_final['result']
data_zhengtai[['fp_times', 'assets', 'crcrd', 'moneyfp', 'trades',
       'banktrade', 'autotrade', 'bankinvestaum', 'goldaum', 'loan',
       'bondaum']].hist(figsize=(40,30),bins=30)
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
#bins = [0,20,30,40,50,60,70]
#data_final_last['age_bins'] = pd.cut(data_final_last['age'],bins,labels=False)
data_final=data_final.join(pd.get_dummies(data_final['marriage'],prefix='marriage'))
#data_final_last=data_final_last.join(pd.get_dummies(data_final_last['rank'],prefix='rank'))
#data_final_last=data_final_last.join(pd.get_dummies(data_final_last['age_bins'],prefix='age'))
data_final.drop('marriage',axis=1,inplace=True)
data_final.isnull().sum()    #有缺失值，用中位数填补
#data_final_last.lnotheraum.fillna(data_final_last.lnotheraum.median(),inplace=True)
datax=data_final.drop('result',axis=1)
datay=pd.DataFrame(data_final['result'])
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

