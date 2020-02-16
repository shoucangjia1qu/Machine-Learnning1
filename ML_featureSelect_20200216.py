# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:56:39 2020

@author: ecupl
"""

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import os
os.chdir(r"D:\mywork\test")

class featureSelect(object):
    #初始化
    def __init__(self, score, categoryMark):
        '''
        score:子集评价的方法，有AIC, BIC, R2, ENTROPY, GINI
        categoryRemark:iterable，特征是否分类变量
        '''
        self.score = score                      #子集评价的方法
        self.catesMark = categoryMark           #是否分类变量的标识
        self.xSelected = 0                      #最终选择的变量
        
        
    #子集搜索前向搜索
    def forwardSearch(self, x, y):
        '''
        基于DataFrame写的前向搜索
        x:2维的DataFrame特征
        y:1维的Series因变量
        '''
        if self.score.upper() == "AIC":
            selected = self.aic_selected(x, y)
        elif self.score.upper() == "BIC":
            selected = self.bic_selected(x, y)
        elif self.score.upper() == "R2":
            selected = self.r2_selected(x, y)
        elif self.score.upper() == "ENTROPY":
            selected = self.entropy_selected(x, y, self.catesMark)
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

    #Entropy评价方法
    def entropy_selected(self, x, y, categoryMark):
        EntTotal = self.__calEntropy(y)
        
        
        
        
        
        return

    #计算信息熵
    def __calEntropy(self, y):
        m = y.size
        yset = np.unique(y)
        ent = 0
        for value in yset:
            ent+=-(sum(y==value)/m)*np.log2(sum(y==value)/m)
        return ent






if __name__ == "__main__":
    raw = pd.read_csv(r'E:\study\L1script\Chapter07 linearmodel\creditcard_exp.csv', skipinitialspace=True)
    exp = raw[raw['avg_exp'].notnull()].copy().iloc[:, 2:].drop('age2',axis=1)
    fs = featureSelect(score='r2')
    x = exp[['Income', 'Age', 'dist_home_val', 'dist_avg_income']]
    y = exp['avg_exp']
    fs.forwardSearch(x, y)
    features = fs.xSelected
    
    
