# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:49:22 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#准备测试数据
Yprob = np.random.random(5000)
Ytrue = np.zeros(5000)
for idx, value in enumerate(Yprob):
    if value>0.5:
        if np.random.random()>=0.2:
            Ytrue[idx] = 1
        else:
            pass
    else:
        if np.random.random()<0.2:
            Ytrue[idx] = 1
        else:
            pass

#sklearn实现
fpr, tpr, thresholds = metrics.roc_curve(Ytrue, Yprob)
auc = metrics.roc_auc_score(Ytrue, Yprob)
plt.plot(fpr, tpr, "b-")
plt.title("Roc_Curve")
plt.show()
print("auc:%.3f"%auc)


#自己编ROC和AUC
#X轴：假正例率（FPR）————所有反例中，预测错误的比率
#Y轴：真正例率（TPR）————所有正例中，预测正确的比率
def roc_curve(y_true, y_prob, n=3):
    """
    Input
    y_true:输入为真实标签，1维数据。
    y_prob:输入为样本打分，1维数据。
    ————————————
    Return
    fpr:假正例率数组。
    tpr:真正例率数组。
    thresholds:每个节点的值。
    """
    y_prob_argsort = y_prob.argsort()
    y_prob_arg_maxtomin = y_prob_argsort[::-1]
    fpr = []
    tpr = []
    thresholds = []
    t_count = sum(y_true==1)                    #正例总数
    f_count = sum(y_true==0)                    #反例总数
    x_count = y_true.shape[0]                   #样本总数
    thre0 = 1                                   #设置初始阈值，可以减少开销
    y_pre = np.zeros(x_count)                   #设置初始按照y_prob的测试结果
    for idx, idx_value in enumerate(y_prob_arg_maxtomin):
        if idx == x_count-1:
            fpr.append(1)
            tpr.append(1)
            thresholds.append(y_prob[idx_value])
        else:
            if idx%n == 0:
                thre1 = y_prob[idx_value]
                y_pre[np.nonzero((y_prob>thre1)&(y_prob<=thre0))[0]] = 1    #将阈值内的预测标签换为1
                tp = sum(y_pre[y_true==1]==1)                               #统计真实标签为1，且预测标签为1的tp数量
                fp = sum(y_pre[y_true==0]==1)                               #统计真实标签为0，且预测标签为1的fp数量
                fpr.append(fp/f_count)
                tpr.append(tp/t_count)
                thresholds.append(thre1)
                thre0 = thre1
            else:
                pass
    return np.array(fpr), np.array(tpr), np.array(thresholds)

def roc_auc(y_true, y_prob):
    t_count = sum(y_true==1)                    #正例总数
    f_count = sum(y_true==0)                    #反例总数
    f_idx = np.nonzero(y_true==0)[0]            #反例的下标
    t_score = y_prob[y_true==1]                 #将正例的得分数组单独划分出来
    lost = 0                                    #初始化损失得分
    for idx in f_idx:
        score = y_prob[idx]
        lost += sum(t_score<score)*1.0          #正例中得分小于反例得分的加1分
        lost += sum(t_score==score)*0.5         #正例中得分等于反例得分的加0.5分
    auc =1 - lost/(t_count*f_count)                #计算AUC
    return auc
        

        
fpr, tpr, thresholds = roc_curve(Ytrue, Yprob)
auc = roc_auc(Ytrue, Yprob)
plt.plot(fpr, tpr, "b-")
plt.title("Roc_Curve")
plt.show()
print("auc:%.3f"%auc)
    
    
    

