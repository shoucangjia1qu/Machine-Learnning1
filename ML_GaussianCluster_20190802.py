# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 21:43:43 2019

@author: ecupl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy

##############西瓜集数据4.0
data = np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],[0.403,0.237],[0.481,0.149],
                 [0.437,0.211],[0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],
                 [0.360,0.370],[0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],[0.748,0.232],
                 [0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],[0.751,0.489],[0.532,0.472],[0.473,0.376],
                 [0.725,0.445],[0.446,0.459]])

#计算高斯分布函数
def Gaussian_multi(x, miu, sigma):
    """
    多元高斯分布的密度函数
    input:
        x:样本集,m*d,其中m为样本数,d为样本维度数
        miu:该高斯分布的均值,1*d维
        sigma:该高斯分布的标准差,在此为d*d的协方差矩阵
    return:
        distributionArr:返回样本的概率分布1D数组
    """
    distributionArr = np.exp(-0.5*np.sum(np.multiply(np.dot(x-miu, np.linalg.inv(sigma)), x-miu), axis=1))/np.power(2*np.pi, 0.5*d)*np.linalg.det(sigma)**0.5
    return distributionArr


#计算观测值y，高斯分布函数参数条件下，观测来自于第k个高斯分布的概率
def Gama_Prob(x, AlphaArr, MiuArr, SigmaArr):
    global k, m
    """
    计算当观测值已知，是哪个高斯模型产品该观测值的概率
    input:
        x:样本集,m*d,其中m为样本数,d为样本维度数
        AlphaArr:每个高斯模型出现的先验概率,1*k维,k为聚类个数
        MiuArr:每个高斯模型的均值参数,k*d维
        SigmaArr:每个高斯模型的协方差矩阵参数,k*d*d维
    return:
        GamaProbArr:每个样本出现对应每个高斯模型分布概率的矩阵,m*k维
    """
    GamaProbArr = np.zeros((m, k))
    for i in range(k):
        miu = MiuArr[i]
        sigma = SigmaArr[i]
        GamaProbArr[:,i] = Gaussian_multi(x, miu, sigma)
    GamaProbArr = np.multiply(GamaProbArr, AlphaArr)
    SumGamaProb = np.sum(GamaProbArr, axis=1).reshape(-1,1)
    return GamaProbArr/SumGamaProb


#计算似然函数



#更新高斯分布函数参数



#更新每个高斯分布函数的混合系数



#初始化函数参数
def initParas(x, k):
    """
    input:
        x:样本集,m*d,其中m为样本数,d为样本维度数
        k:需要聚类的个数
    return:
        初始化AlphaArr, MiuArr, SigmaArr
    """
    m, d = np.shape(x)
    AlphaArr0 = np.ones((1,k))/k
    MiuArr0 = x[np.random.randint(0, m, k)]
    #在这里固定好了
    MiuArr0 = x[[5,21,26]]
    SigmaArr0 = np.array([[[0.1,0],[0,0.1]]]*k)
    return AlphaArr0, MiuArr0, SigmaArr0


#按照最终结果划分类型



#画图



#外循环：判断是否符合停止条件



#内循环：计算和更新各种参数





