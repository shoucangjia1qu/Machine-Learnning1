# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:04:34 2018

@author: ecupl
"""

import numpy as np
import pandas as pd
import os,copy
import matplotlib.pyplot as plt

#######################
#                     #
#     HMM最优路径      #
#                     #
#######################
os.chdir(r"D:\mywork\test")
'''1、近似算法'''
#初始准备
startP = np.array([[0.63,0.17,0.20]])
transformP = np.array([[0.5,0.375,0.125],[0.25,0.125,0.625],[0.25,0.375,0.375]])
observeP = np.array([[0.6,0.2,0.15,0.05],[0.25,0.25,0.25,0.25],[0.05,0.1,0.35,0.5]])
stateArray = ["晴天","阴天","雨天"]
observeArray = ["干旱","干燥","湿润","潮湿"]

#开始循环计算
stateResult = []
observeResult = ["干旱","干燥","潮湿"]
for idx in range(len(observeResult)):
    stateDict = {}
    if idx==0:
        observeIdx = observeArray.index(observeResult[idx])
        stateP = np.multiply(startP,observeP[:,observeIdx])
        state = stateArray[np.argmax(stateP)]
    else:
        for i in stateResult[idx-1].values():
            stateForward = i
        observeIdx = observeArray.index(observeResult[idx])
        stateP = np.multiply(np.dot(stateForward,transformP),observeP[:,observeIdx])
        state = stateArray[np.argmax(stateP)]
    stateDict[state] = stateP
    stateResult.append(stateDict)

'''2、Vertibi维特比算法'''
def vertibi(observeResult,startP,transformP,observeP,stateArray,observeArray):
    '''
    observeResult:实际观测结果
    startP:初始概率
    transformP:状态转移概率
    observeP:观测发射概率
    stateArray:状态序列
    observeArray:观测序列
    '''
    stateP = []         #初始化状态概率
    state = []          #初始化状态情况
    for idx in range(len(observeResult)):
        statePdict = dict()
        statedict = dict()
        if idx==0:
            observeIdx = observeArray.index(observeResult[idx])
            tempstateP = np.multiply(startP,observeP[:,observeIdx]).reshape(-1)
        else:
            stateForwardP = np.array(list(stateP[idx-1].values())).reshape(-1,1)
            observeIdx = observeArray.index(observeResult[idx])
            statePro = np.multiply(stateForwardP,transformP).max(axis=0)
            tempstateP = np.multiply(statePro,observeP[:,observeIdx])
            print(statePro,observeP[:,observeIdx])
        for i in range(len(tempstateP)):
            statePdict[stateArray[i]] = tempstateP[i]
        statedict[observeResult[idx]] = stateArray[np.argmax(tempstateP)]
        stateP.append(statePdict),state.append(statedict)
    print(state,stateP)


#######################
#                     #
#     HMM词性标注      #
#                     #
#######################
import jieba
'''不带词性'''
sen = jieba.cut("把这篇报道修改一下")
for x in sen:
    print(x)
    
'''带词性'''
import jieba.posseg
sen2 = jieba.posseg.cut("把这篇报道修改一下")
for i in sen2:
    print(i.word,i.flag)







