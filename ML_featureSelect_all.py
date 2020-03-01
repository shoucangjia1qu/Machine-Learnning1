# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:56:39 2020

@author: ecupl
"""

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.api import OLS
from sklearn.feature_selection import f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression
from scipy import stats
import os
os.chdir(r"D:\mywork\test\02code_MLfeature")

import woeself.dataprocessing as wdp
import ML_reliefself as ref

#子集搜索前向搜索
class forwardSearch(object):
    #初始化
    def __init__(self, scoring):
        '''
        Parameters
        ----------
        scoring: 子集评价的方法，有AIC, BIC, R2
        '''
        self.scoring = scoring                      #子集评价的方法
        self.xSelected = 0                          #最终选择的变量
    
    #训练
    def fit(self, x, y):
        '''
        基于DataFrame写的前向搜索
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Results
        -------
        xSelected: list, [(score1, colName1), (score2, colName2), ... (scoreX, colNameX)]
        '''
        if self.scoring.upper() == "AIC":
            selected = self.aic_selected(x, y)
        elif self.scoring.upper() == "BIC":
            selected = self.bic_selected(x, y)
        elif self.scoring.upper() == "R2":
            selected = self.r2_selected(x, y)
        else:
            pass
        FF = "{} ~ {}".format(y.name, " + ".join(selected))
        print("Final formula is %s"%FF)
        self.xSelected = selected
        return
    
    #AIC评价方法
    def aic_selected(self, x, y)->list:
        x_remaining = set(x.columns)
        response = y.name
        data = pd.concat([x, y], axis=1)
        selected = []
        best_score = np.inf; current_score = np.inf
        while x_remaining:
            aic_xcandidates = []
            for col in x_remaining:
                formula = "{} ~ {}".format(response, " + ".join(selected+[col]))
                results = ols(formula=formula, data=data).fit()
                aic = results.aic
                aic_xcandidates.append((aic, col))
            aic_xcandidates.sort(reverse=True)
            current_score, current_col = aic_xcandidates.pop()
            if current_score < best_score:
                best_score = current_score
                x_remaining.remove(current_col)
                selected.append(current_col)
                print("AIC is %f, "%best_score,"FEATURE is %s!"%current_col)
            else:
                print("AIC Select is over!")
                break
        return selected
                
    #BIC评价方法
    def bic_selected(self, x, y)->list:
        x_remaining = set(x.columns)
        response = y.name
        data = pd.concat([x, y], axis=1)
        selected = []
        best_score = np.inf; current_score = np.inf
        while x_remaining:
            bic_xcandidates = []
            for col in x_remaining:
                formula = "{} ~ {}".format(response, " + ".join(selected+[col]))
                results = ols(formula=formula, data=data).fit()
                bic = results.bic
                bic_xcandidates.append((bic, col))
            bic_xcandidates.sort(reverse=True)
            current_score, current_col = bic_xcandidates.pop()
            if current_score < best_score:
                best_score = current_score
                x_remaining.remove(current_col)
                selected.append(current_col)
                print("BIC is %f, "%best_score,"FEATURE is %s!"%current_col)
            else:
                print("BIC Select is over!")
                break
        return selected

    #R2评价方法
    def r2_selected(self, x, y)->list:
        x_remaining = set(x.columns)
        response = y.name
        data = pd.concat([x, y], axis=1)
        selected = []
        best_score = 0; current_score = 0
        while x_remaining:
            r2_xcandidates = []
            for col in x_remaining:
                formula = "{} ~ {}".format(response, " + ".join(selected+[col]))
                results = ols(formula=formula, data=data).fit()
                r2 = results.rsquared
                r2_xcandidates.append((r2, col))
            r2_xcandidates.sort()
            current_score, current_col = r2_xcandidates.pop()
            if current_score > best_score:
                best_score = current_score
                x_remaining.remove(current_col)
                selected.append(current_col)
                print("R2 is %f, "%best_score,"FEATURE is %s!"%current_col)
            else:
                print("R2 Select is over!")
                break
        return selected



#%%
#子集搜索过滤式选择
class filterSelect(object):
    #初始化
    def __init__(self, scoring, categoryMarks=[], api='sklearn'):
        '''
        Parameters
        ----------
        scoring: 子集评价的方法;
                    分类: chi2, fclassify, miclassify, entropy, gini, 
                            relief(目前可支持多分类), iv(仅支持二分类);
                    回归: fregression, miregression;
        categoryMarks: iterable，每个特征是否分类变量，1-是，0-否，默认为空列表
        api: 计算统计量的方法是直接调用sklearn的，还是self的，默认sklearn
        '''
        self.scoring = scoring                      #子集评价的方法
        self.catesMarks = categoryMarks             #每个特征是否分类变量
        self.api = api                              #使用的api类型
        self.xSelected = 0                          #最终选择的变量
    
    #训练
    def fit(self, x, y):
        '''
        过滤式选择
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Results
        -------
        xSelected: list
        '''
        if self.scoring.lower() == "fclassify":
            self.xSelected = self.fclassify_selected(x, y)
        elif self.scoring.lower() == "fregression":
            self.xSelected = self.fregression_selected(x, y)
        elif self.scoring.lower() == "chi2":
            self.xSelected = self.chi2_selected(x, y)
        elif self.scoring.lower() == "entropy":
            self.xSelected = self.entropy_selected(x, y)
        elif self.scoring.lower() == "gini":
            self.xSelected = self.gini_selected(x, y)
        elif self.scoring.lower() == "miclassify":
            self.xSelected = self.miclassify_selected(x, y)
        elif self.scoring.lower() == "miregression":
            self.xSelected = self.miregression_selected(x, y)
        elif self.scoring.lower() == "iv":
            self.xSelected = self.iv_selected(x, y)
        elif self.scoring.lower() == "relief":
            self.xSelected = self.relief_selected(x, y)
            
            
            
        return
    
    #fclassify方法，F检验-方差分析方法，用于分类模型
    def fclassify_selected(self, x, y):
        '''
        用于分类模型，方差分析方法
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Returns
        -------
        xSelected: list, [(F-statistic, pValue, colName),......]
        
        Others
        ------
        api为'self'时，调用自编的方差分析函数，逐个计算F统计量和P值，并打印结果矩阵
        '''
        x_remaining = list(x.columns)
        selected = []
        if self.api == "sklearn":
            F_statistic, pValue = f_classif(x, y)
            for fs, pv, col in zip(F_statistic, pValue, x_remaining):
                selected.append((fs, pv, col))
        elif self.api == "self":
            for feature in x_remaining:
                fs, pv = self.__calAnova(x[feature], y)
                selected.append((fs, pv, feature))
        else:
            raise ValueError("api must be 'sklearn' or 'self' !")
        selected.sort(reverse=True)
        return selected
    
    #自编方差分析计算函数
    def __calAnova(self, x, y):
        """
        Parameters
        ----------
        x:变量[1D]array
        y:实际标签[1D]array
        
        Returns
        -------
        Fstatistic: F统计量
        pValue: F统计量对应的P值
        """
        yValues = np.unique(y); xbar = x.mean()
        nsamples =y.size; nlabels = yValues.size
        ximean = []; xicount = []                               #每类样本的均值和数量列表
        #1 计算自由度
        dfList = [nlabels-1, nsamples-nlabels, nsamples-1]      #(组间，组内，合计)
        ##2.1 组内离差平方和
        SSList = []
        SSw = 0
        for value in yValues:
            xi = x[y==value]
            xicount.append(len(xi))         #每类的数量
            xmean = xi.mean()
            ximean.append(xmean)            #每类的均值
            SSw += np.power((xi-xmean), 2).sum()
        ##2.2 组间离差平方和
        SSb = np.dot(np.power((np.array(ximean)-xbar), 2), xicount)
        ##2.3 合计离差平方和
        SSt = SSw + SSb
        SSList = [SSb, SSw, SSt]
        #3 计算均方
        MSList = [SSb/dfList[0], SSw/dfList[1]]
        #4 计算F值和P值
        Fstatistic = MSList[0]/MSList[1]
        pValue = stats.f.sf(Fstatistic, dfList[0], dfList[1])
        #5 打印df
        df = pd.DataFrame(index=['组间', '组内', '合计'], columns=['自由度', '离差平方和', '均方', 'F值', 'P值'])
        df.iloc[:,0] = dfList; df.iloc[:,1] = SSList; df.iloc[:2,2] = MSList
        df.iloc[0,3] = Fstatistic; df.iloc[0,4] = pValue
        print(df,'\n-------------------------\n')
        return Fstatistic, pValue
    
    #fregression方法，F检验-单变量回归显著性检验，用于回归模型
    def fregression_selected(self, x, y):
        '''
        用于回归模型，单变量回归检验模型显著性。
        在一元线性回归中，模型F统计量=系数t统计量，模型R2=相关系数的平方。
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Returns
        -------
        xSelected: list, [(F-statistic, pValue, colName),......]
        
        Others
        ------
        api为'self'时，根据单变量和Y值之间的相关系数，求得R2，再计算F统计量和P值
        '''
        x_remaining = list(x.columns)
        selected = []
        if self.api == "sklearn":
            F_statistic, pValue = f_regression(x, y)
            for fs, pv, col in zip(F_statistic, pValue, x_remaining):
                selected.append((fs, pv, col))
        elif self.api == "statsmodels":
            intercept = np.ones((y.size,1))
            for feature in x_remaining:
                lms = OLS(y, np.hstack([intercept, x[feature].values.reshape(-1,1)])).fit()
                selected.append((lms.fvalue, lms.f_pvalue, feature))
        elif self.api == "self":
            for feature in x_remaining:
                corr = np.corrcoef(x[feature], y)[1,0]
                r2 = corr**2
                fs = (y.size-2.)*r2/(1.-r2)
                pv = stats.f.sf(fs, 1, y.size-2)
                selected.append((fs, pv, feature))
        else:
            raise ValueError("api must be 'sklearn' or 'statsmodels' or 'self' !")
        selected.sort(reverse=True)
        return selected
    
    #chi2方法，卡方检验
    def chi2_selected(self, x, y):
        '''
        用于分类模型。
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Returns
        -------
        xSelected: list, [(chiq, pValue, colName),......]
        
        Others
        ------
        *api为'sklearn'时，用sklearn.feature_selection.chi2，直接计算每种分类下的特征均值，
         适用于0-1或者类似频率等特征。
        *api为'stats'时，对每个特征的特征值进行观测值和预测值的计算，仅适用于分类变量。卡方值会大于sklearn。
        *api为'self'时，是自己编写的求卡方值和P值。
        '''
        x_remaining = list(x.columns)
        selected = []
        if self.api == "sklearn":
            chiq, pValue = chi2(x, y)
            for cq, pv, col in zip(chiq, pValue, x_remaining):
                selected.append((cq, pv, col))
        elif self.api == "stats":
            for feature in x_remaining:
                cross_table = pd.crosstab(columns = y, index = x[feature])
                chiq, pValue, df, expected_freq= stats.chi2_contingency(cross_table)
                selected.append((chiq, pValue, feature))
        elif self.api == "self":
            for feature in x_remaining:
                chiq, pValue = self.__calChi2(x[feature], y)
                selected.append((chiq, pValue, feature))
        else:
            raise ValueError("api must be 'sklearn' or 'stats' or 'self' !")
        selected.sort(reverse=True)
        return selected
    
    #自编卡方检验函数
    def __calChi2(self, x, y):
        '''
        Parameters
        ----------
        x:变量[1D]array
        y:实际标签[1D]array
        
        Returns
        -------
        chi2Value: 卡方值
        pValue: 对卡方值进行T检验的P值
        '''
        nsamples = y.size
        xValues = np.unique(x)
        yValues = np.unique(y)
        #y的分布
        PyValues = [sum(y==yvalue)/nsamples for yvalue in yValues]
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
            pValue = stats.chi2.sf(chi2Value, dfreedom+1)
        else:
            pValue = stats.chi2.sf(chi2Value, dfreedom)
        return chi2Value, pValue
    
    #Entropy信息增益评价方法
    def entropy_selected(self, x, y):
        '''
        信息增益选取特征，适用分类模型。
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Returns
        -------
        xSelected: list, [(entropyGain, colName),......]
        '''
        EntTotal = self.__calEntropy(y)
        nsamples = y.size
        x_remaining = list(x.columns)
        selected = []
        for xfeature, mark in zip(x_remaining, self.catesMarks):
            xiSet = np.sort(np.unique(x[xfeature]))
            if mark == 1:
                p = [sum(x[xfeature]==i)/nsamples for i in xiSet]
                EntSub = [self.__calEntropy(y[(x[xfeature]==i).nonzero()[0]]) for i in xiSet]
                Gain = EntTotal - np.dot(p, EntSub)
            elif mark == 0:
                minEnt = np.inf
                for idx, value in enumerate(xiSet[:-1]):
                    thres = (value + xiSet[idx+1])/2
                    p = [sum(x[xfeature]<thres)/nsamples, sum(x[xfeature]>thres)/nsamples]
                    EntSub = [self.__calEntropy(y[(x[xfeature]<thres).nonzero()[0]]), self.__calEntropy(y[(x[xfeature]>thres).nonzero()[0]])]
                    EntSS = np.dot(p, EntSub)
                    if EntSS < minEnt:
                        minEnt = EntSS
                        bestthres = thres
#                print("{}:{}".format(xfeature, bestthres))
                Gain = EntTotal - minEnt
            else:
                raise ValueError("categoryMark must be 1 or 0 !")
            selected.append((Gain, xfeature))
            selected.sort(reverse=True)
        return selected

    #计算信息熵函数
    def __calEntropy(self, y):
        """
        Parameters
        ----------
        x:变量[1D]array
        y:实际标签[1D]array
        
        Returns
        -------
        ent: 信息熵
        """
        m = y.size
        yset = np.unique(y)
        ent = 0
        for value in yset:
            ent+=-(sum(y==value)/m)*np.log2(sum(y==value)/m)
        return ent
    
    #GINI基尼指数的评价方法
    def gini_selected(self, x, y):
        '''
        基尼指数选取特征，适用分类模型。
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Returns
        -------
        xSelected: list, [(Gini, colName),......]
        '''
        nsamples = y.size
        x_remaining = list(x.columns)
        selected = []
        for xfeature, mark in zip(x_remaining, self.catesMarks):
            xiSet = np.sort(np.unique(x[xfeature]))
            minGini = np.inf
            if mark == 1:
                for idx, value in enumerate(xiSet):
                    thres = value
                    p = [sum(x[xfeature]==thres)/nsamples, sum(x[xfeature]!=thres)/nsamples]
                    GiniSub = [self.__calGini(y[(x[xfeature]==thres).nonzero()[0]]), self.__calGini(y[(x[xfeature]!=thres).nonzero()[0]])]
                    GiniSS = np.dot(p, GiniSub)
                    print("变量：{}；取值：{}；尼基指数：{}".format(xfeature,thres,GiniSS))
                    if GiniSS < minGini:
                        minGini = GiniSS
                        bestthres = thres
            elif mark == 0:
                for idx, value in enumerate(xiSet[:-1]):    
                    thres = (value + xiSet[idx+1])/2
                    p = [sum(x[xfeature]<thres)/nsamples, sum(x[xfeature]>thres)/nsamples]
                    GiniSub = [self.__calGini(y[(x[xfeature]<thres).nonzero()[0]]), self.__calGini(y[(x[xfeature]>thres).nonzero()[0]])]
                    GiniSS = np.dot(p, GiniSub)
                    print("变量：{}；取值：{}；尼基指数：{}".format(xfeature,thres,GiniSS))
                    if GiniSS < minGini:
                        minGini = GiniSS
                        bestthres = thres
            else:
                raise ValueError("categoryMark must be 1 or 0 !")
            print("最小变量{}:{} ~ {}\n".format(xfeature, bestthres, minGini))
            selected.append((minGini, xfeature))
        selected.sort()
        return selected
    
    #计算GINI指数函数
    def __calGini(self, y):
        """
        Parameters
        ----------
        x:变量[1D]array
        y:实际标签[1D]array
        
        Returns
        -------
        gini: 基尼指数
        """
        m = y.size
        yset = np.unique(y)
        gini = 1
        for value in yset:
            gini -= (sum(y==value)/m)**2
        return gini
    
    #miclassify方法，基于KNN的互信息度量，用于分类模型
    def miclassify_selected(self, x, y):
        '''
        用于分类模型，方差分析方法
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Returns
        -------
        xSelected: list, [(Mi, colName),......]
        
        Others
        ------
        方法有待推导
        '''
        x_remaining = list(x.columns)
        selected = []
        if self.api == "sklearn":
            Mi = mutual_info_classif(x, y)
            for m, col in zip(Mi, x_remaining):
                selected.append((m, col))
        elif self.api == "self":
            pass
        else:
            raise ValueError("api must be 'sklearn' or 'self' !")
        selected.sort(reverse=True)
        return selected
    
    #miregression方法，基于KNN的互信息度量，用于回归模型
    def miregression_selected(self, x, y):
        '''
        用于分类模型，方差分析方法
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Returns
        -------
        xSelected: list, [(Mi, colName),......]
        
        Others
        ------
        方法有待推导
        '''
        x_remaining = list(x.columns)
        selected = []
        if self.api == "sklearn":
            Mi = mutual_info_regression(x, y)
            for m, col in zip(Mi, x_remaining):
                selected.append((m, col))
        elif self.api == "self":
            pass
        else:
            raise ValueError("api must be 'sklearn' or 'self' !")
        selected.sort(reverse=True)
        return selected

    #IV值
    def iv_selected(self, x, y):
        '''
        用于分类模型，方差分析方法
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Returns
        -------
        xSelected: list, [(IV, colName),......]
        
        Others
        ------
        调用了自己写的woeself的函数
        '''
        bdtt = sum(y)
        gdtt = y.size - bdtt
        x_remaining = list(x.columns)
        selected = []
        for xfeature in x_remaining:
            IV = 0
            data_xfeature = x[xfeature]
            for value in np.unique(data_xfeature):
                detail = wdp.calIV(data_xfeature[data_xfeature==value], y[data_xfeature==value], bdtt, gdtt)
                IV += detail.get('iv_sub')
            selected.append((IV, xfeature))
        selected.sort(reverse = True)
        return selected
    
    #Relief统计量
    def relief_selected(self, x, y):
        '''
        用于分类模型，方差分析方法
        Parameters
        ----------
        x: 2维的DataFrame特征
        y: 1维的Series因变量
        
        Returns
        -------
        xSelected: list, [(IV, colName),......]
        
        Others
        ------
        调用了自己写的relief类
        '''
        x_remaining = list(x.columns)
        selected = []
        relief_clf = ref.Relevant_feature()
        relief_clf.train(x, y, self.catesMarks)
        reliefW = relief_clf.W
        for colName, wi in zip(x_remaining, reliefW):
            selected.append((wi,colName))
        selected.sort(reverse = True)
        return selected
    
    
#%%包裹式选择
##递归式特征消除(recursive feature elimination)
from sklearn.svm import SVC
from sklearn.datasets import load_iris, make_classification
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold
#鸢尾花数据集
x, y = load_iris(return_X_y=True)
#训练
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=2, step=1, verbose=32)
'''
n_features_to_select ：选出的特征整数时为选出特征的个数，None时选取一半
step ： 整数时，每次去除的特征个数，小于1时，每次去除权重最小的特征
'''
rfe.fit(x, y)
newx = rfe.transform(x)             #递归消除特征后的数据集
#查看结果
print(rfe.ranking_)                 #特征重要性排名
print(rfe.support_)                 #是否保留这个特征

##递归式特征消除CV(recursive feature elimination CV)
##RFECV，鸢尾花
rfecv = RFECV(estimator=svc, step=1, min_features_to_select=2, cv=3, scoring='accuracy', verbose=32)
rfecv.fit(x, y)
rfecv.grid_scores_
rfecv.get_support()
##RFECV，其他数据集
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)
rfecv = RFECV(estimator=svc, step=1, min_features_to_select=1, cv=StratifiedKFold(6), scoring='accuracy', verbose=32)
'''
min_features_to_select ：每轮迭代保留的最小特征个数
step ：整数时，每次去除的特征个数，小于1时，每次去除权重最小的特征
cv ：几折交叉验证
scoring ：交叉验证的评价分数
'''
rfecv.fit(X, y)
rfecv.grid_scores_
rfecv.get_support()
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


#%%测试

if __name__ == "__main__":
    ###########向前法###########
    #数据准备
    raw = pd.read_csv(r'E:\study\L1script\Chapter07 linearmodel\creditcard_exp.csv', skipinitialspace=True)
    exp = raw[raw['avg_exp'].notnull()].copy().iloc[:, 2:].drop('age2',axis=1)
    #实例化
    x1 = exp[['Income', 'Age', 'dist_home_val', 'dist_avg_income']]
    y1 = exp['avg_exp']
    fs = forwardSearch(scoring='aic')
    fs.fit(x1, y1)
    print(fs.xSelected)
    
    ###########过滤式选择#########
    import woe.feature_process
    fs = filterSelect('relief', categoryMarks=[1,1,1,1,1,1,0,0])
    fs.fit(X, Y)
    a1 = fs.xSelected
     
    fs = filterSelect('miclassify')
    fs.fit(X, Y)
    a2 = fs.xSelected
    print(a2)
      
    fs = filterSelect('entropy', categoryMarks=[1,1,1,1,1,1,0,0])
    fs.fit(X, Y)
    a3 = fs.xSelected
    
    #西瓜集数据集
    dataSet = [
            ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
            ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
            ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
            ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
            ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
        ]
    #特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
    #整理出数据集和标签
    X = np.array(dataSet)[:,:8]
    Y = np.array(dataSet)[:,8]
    
    #对X进行编码
    from sklearn.preprocessing import OrdinalEncoder
    oriencode = OrdinalEncoder(categories='auto')
    oriencode.fit(X[:,:6])
    Xdata=oriencode.transform(X[:,:6])           #编码后的数据
    print(oriencode.categories_)                       #查看分类标签
    Xdata=np.hstack((Xdata,X[:,6:].astype(float)))
    
    #对Y进行编码
    from sklearn.preprocessing import LabelEncoder
    labelencode = LabelEncoder()
    labelencode.fit(Y)
    Ylabel=labelencode.transform(Y)       #得到切分后的数据
    labelencode.classes_                        #查看分类标签
    labelencode.inverse_transform(Ylabel)    #还原编码前数据
    X = pd.DataFrame(Xdata, columns=labels)
    Y = pd.Series(Ylabel)

    ##########包裹式选择##########


