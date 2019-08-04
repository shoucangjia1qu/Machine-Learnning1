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
    distributionArr = np.exp(-0.5*np.sum(np.multiply(np.dot(x-miu, np.linalg.pinv(sigma)), x-miu), axis=1))/np.power(2*np.pi, 0.5*d)*np.linalg.det(sigma)**0.5
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
    return (GamaProbArr/SumGamaProb).round(4)


#计算似然函数



#更新高斯分布函数参数
def updateParas(x, GamaProbArr):
    global k, d
    """
    更新高斯分布函数的参数，包括均值和协方差矩阵
    input:
        x:样本集,m*d,其中m为样本数,d为样本维度数
        GamaProbArr:高斯分布函数的后验概率,m*k维
    return:
        newMiuArr:更新后的高斯分布函数的均值,k*d维
        newSigmaArr:更新后的高斯分布的协方差矩阵,k*d*d维
        newAlphaArr:更新后的高斯模型的混合系数,1*k维
    """
    SumGamaProb = np.sum(GamaProbArr, axis=0)
    newMiuArr = np.zeros((k,d))
    newSigmaArr = np.zeros((k,d,d))
    for i in range(k):
        Gama = GamaProbArr[:,i].reshape(-1,1)
        #更新均值
        newMiu = np.sum(np.multiply(Gama, x), axis=0)/SumGamaProb[i]
        newMiuArr[i] = newMiu
        #更新协方差矩阵
        newSigma = np.dot(np.multiply(x-newMiu, Gama).T, x-newMiu)/SumGamaProb[i]
        newSigmaArr[i] = newSigma
    newAlphaArr = SumGamaProb.reshape(1,-1)/m
    return newMiuArr.round(4), newSigmaArr.round(4), newAlphaArr.round(4)


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
    SigmaArr0 = np.array([(np.eye(d)*0.1).tolist()]*k)
    return AlphaArr0, MiuArr0, SigmaArr0


#按照最终结果划分类型
#ClusterLabel = np.argmax(GamaProbArr, axis=1)


#计算X*Y的高斯分布
def calXYZ(x, MiuArr, SigmaArr):
    global m
    """
    画等高图需要计算X,Y,Z
    input:
        x:样本集,m*d,其中m为样本数,d为样本维度数
        MiuArr, SigmaArr:高斯分布函数的参数
    return:
        xgrid:x的网格坐标
        ygrid:y的网格坐标
        zgrid:(x,y)网格坐标上高斯分布函数的概率
    """
    x1 = np.copy(x[:,0])
    x1.sort()
    y1 = np.copy(x[:,1])
    y1.sort()
    x2,y2 = np.meshgrid(x1,y1)  # 获得网格坐标矩阵
    Gp = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            xi = x2[i,j]
            yi = y2[i,j]
            data = np.array([xi,yi])
            miuList=[]
            for miu, sigma in zip(MiuArr, SigmaArr):   
                p = np.exp(-0.5*np.dot(np.dot((data-miu).reshape(1,-1), np.linalg.inv(sigma)), (data-miu).reshape(-1,1)))/np.power(2*np.pi, 0.5*d)*np.linalg.det(sigma)**0.5
                miuList.append(p)
            Gp[i,j] = max(miuList)
    return x2, y2, Gp
    

#画图
def drawPics(x, MiuArr, SigmaArr, Clusters):
    """
    画图，不同聚类类别的点分布，等高图
    input:
        x:样本集,m*d,其中m为样本数,d为样本维度数
        MiuArr, SigmaArr:高斯分布函数的参数
        Clusters:聚类结果
    out:
        散点图+等高分布图
    """
    plt.figure(figsize=(10,6))
    xgrid, ygrid, zgrid = calXYZ(x, MiuArr, SigmaArr)
    c=plt.contour(xgrid,ygrid,zgrid,6,colors='black')
    plt.contourf(xgrid,ygrid,zgrid,6,cmap=plt.cm.Blues,alpha=0.5)
    #plt.clabel(c,inline=True,fontsize=10)
    for i in range(k):
        xi = x[Clusters==i,0]
        yi = x[Clusters==i,1]
        plt.scatter(xi, yi)
        plt.scatter(MiuArr[i,0], MiuArr[i,1], c='r', linewidths=5, marker='D')
    plt.show()
    return


#训练：判断是否符合停止条件
def train(x, k, iters):
    """
    循环迭代
    input:
        x:样本集,m*d,其中m为样本数,d为样本维度数
        k:聚类个数
        iters:迭代次数
    return:
        ClusterLabel:最终的聚类结果
    """
    #初始化参数
    AlphaArr0, MiuArr0, SigmaArr0 = initParas(x, k)
    LLvalue0 = 0                #初始似然函数值
    LLvalueList = []            #最大似然值列表
    for i in range(iters):
        #计算高斯分布模型的后验概率，也就是已知观测下来自于第k个高斯分布函数的概率
        GamaProbArr = Gama_Prob(x, AlphaArr0, MiuArr0, SigmaArr0)
        #计算聚类结果
        ClusterLabel = np.argmax(GamaProbArr, axis=1)
        #画分布图
        drawPics(x, MiuArr0, SigmaArr0, ClusterLabel)
        #计算似然函数，并判断是否继续更新
        
        #继续迭代，更新函数参数
        MiuArr1, SigmaArr1, AlphaArr1 = updateParas(x, GamaProbArr)
        MiuArr0 = np.copy(MiuArr1)
        SigmaArr0 = np.copy(SigmaArr1) 
        AlphaArr0 = np.copy(AlphaArr1)































