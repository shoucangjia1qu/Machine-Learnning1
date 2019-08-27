# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:10:33 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir(r"D:\mywork\test")

with open(r"D:\mywork\test\UCI_data\iris.data") as f:
    data = f.readlines()
trainSet = np.array([row.split(',') for row in data[:-1]])
trainSet = trainSet[:,:-1].astype('float')

#主成分分析
class PrincipalComponentAnalysis(object):
    #1、类属性
    def __init__(self):
        self.X = 0              #原数据
        self.tranX = 0          #降维后的数据
        self.lambdas = 0        #特征向量
        self.V = 0              #特征矩阵
        self.n_components = 0   #特征值个数
        self.pre_components = 0 #主成分百分比
    
    #2、中心标准化（必须做，因为前提就是中心化，各列和分别为0）
    def normnalize(self, x):
        """
        中心标准化
        Input:
            输入m*d的数据
        return:
            返回标准化数据
        """
        return (x - x.mean(axis=0))/x.std(axis=0)
    
    #3、根据主成分百分比选取前N个主成分
    def select_ncomponents(self, lambdas, pre_components):
        pre_lambdas = lambdas/lambdas.sum()
        sort_idx = np.argsort(pre_lambdas)
        percent = 1
        pre_index = 0
        for index, idxValue in enumerate(sort_idx):
            percent -= pre_lambdas[idxValue]
            if percent > pre_components:
                continue
            else:
                pre_index = index
                break
        lambdas_idx = sort_idx[pre_index:]
        re_lambdas_idx = lambdas_idx[::-1]
        return re_lambdas_idx
    
    #4、训练
    def PCAtrain(self, x, method='svd', pre_components=0.95):
        self.X = x
        normX = self.normnalize(x)
        if method == 'svd':
            U, S, Vt = np.linalg.svd(normX)
            V = Vt.T
            lambdas = np.power(S, 2)
        else:
            XTX = np.dot(normX.T, normX)
            lambdas, V = np.linalg.eigh(XTX)
        target_idx = self.select_ncomponents(lambdas, pre_components)
        self.n_components = len(target_idx)
        self.lambdas = lambdas[target_idx]
        self.pre_components = (lambdas/lambdas.sum())[target_idx]
        self.V = V[:,target_idx]
        self.tranX = np.dot(normX, self.V)
        return
            
#训练
if __name__ == "__main__":
    pca_myself = PrincipalComponentAnalysis()
    pca_myself.PCAtrain(trainSet, pre_components=0.95)
    #和Sklearn中的PCA符号不太一样。
    from sklearn.decomposition import PCA
    pca = PCA()
    normX = pca_myself.normnalize(trainSet)
    #直接pca.fit_transform(normX)
    pca.fit(normX)
    pca.transform(normX)
    
        
        
        
        
        
        
        
        
        

