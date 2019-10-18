# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:22:27 2019

@author: ecupl
"""

import numpy as np
import pandas as pd
from scipy import stats     #求众数
import os, copy

os.chdir("D:\\mywork")

dataSet = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])

###1、构造kd树
#1.1 选择维度
def select_fea(x, n_iters):
    """
    第一种方法：按顺序选择需要切分的维度
    第二种方法：计算每个维度的方差，选取方差最大的作为切分的维度
    本代码仅实现第一种方法。
    """
    m, d = np.shape(x)
    return np.mod(n_iters, d)

#1.2根据维度选择切分的特征值
def select_value(x, d):
    """
    为了二分的平衡性，考虑选择属性d中的中位数。
    以及返回在划分平面上的样本，小于和大于划分值的样本
    """
    X_d = x[:,d]
    value = np.median(X_d)
    x_idx = np.nonzero(X_d == value)[0]
    left_idx = np.nonzero(X_d < value)[0]
    right_idx = np.nonzero(X_d > value)[0]
    return value, x_idx, left_idx, right_idx

#1.3判断是否继续递归划分
def judge_stop(x_idx):
    """
    若子节点小于等于1个样本，则停止划分
    """
    if len(x_idx) <= 1:
        return 1
    else:
        return 0 

#1.4赋值每个子结点
def get_node(x, n_iters):
    node = dict()       #生成空的节点字典
    d = select_fea(x, n_iters)
    node['d'] = d       #赋值划分维度
    value, x_idx, left_idx, right_idx = select_value(x, d)
    node['d_value'] = value; node['d_set'] = x[x_idx]
    #判断是否迭代,先递归左边，再递归右边
    #左边
    left_stop = judge_stop(left_idx)
    left_iters = copy.deepcopy(n_iters)
    if left_stop == 1:
        node['left_set'] = x[left_idx]
    else:
        left_x = x[left_idx]
        left_iters += 1
        node['left_set'] = get_node(left_x, left_iters)
    #右边
    right_stop = judge_stop(right_idx)
    right_iters = copy.deepcopy(n_iters)
    if right_stop == 1:
        node['right_set'] = x[right_idx]
    else:
        right_x = x[right_idx]
        right_iters += 1
        node['right_set'] = get_node(right_x, right_iters)
    return node

#1.5构造kd树
def build_kdTree(x):
    n_iters = 0
    kd_Tree = get_node(x, n_iters)
    return kd_Tree
    
#构造kd树
kd_Tree = build_kdTree(dataSet)
    
    
    
    
    


