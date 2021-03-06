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


###2、搜索kd树
#2.1递归寻找近似点
def find_similar(target, kd_Tree):
    kdTree = copy.deepcopy(kd_Tree)
    d = kdTree['d']
    d_value = kdTree['d_value']
    target_dvalue = target[d]
    if target_dvalue <= d_value:
        left_tree = kdTree['left_set']
        if isinstance(left_tree, dict):
            kdTree['left_set'] = find_similar(target, left_tree)
        else:
            min_dist = np.linalg.norm(target-left_tree)             #计算目标样本与近似点的距离
            min_point = left_tree
            print("直接的最近距离：",min_dist)
            print("直接的最近样本点：",min_point)
            #计算划分平面上的点到目标样本的距离
            if len(kdTree['d_set']) >= 1:
                d_dist = min(np.linalg.norm((target - kdTree['d_set']), axis=1))
                d_idx = np.argmin(np.linalg.norm((target - kdTree['d_set']), axis=1))
                d_point = kdTree['d_set'][d_idx]
                print("分类平面上的最近距离：",d_dist)
                print("分类平面上的最近样本点：",d_point)
                if d_dist < min_dist:
                    min_dist = copy.deepcopy(d_dist)
                    min_point = copy.deepcopy(d_point)
            #计算右边的子集，获取最近点
            if abs(target[kdTree['d']] - kdTree['d_value']) <= min_dist:
                right_tree = kdTree['right_set']
                #右边子树寻找最小点，递归
                print("    新的递归*********")
                right_point = find_minpoint(target, right_tree)
                print("    新的递归结束*********")
                right_dist = np.linalg.norm(target-right_point)
                print("分类另一面的最近距离：",right_dist)
                print("分类另一面的最近样本点：",right_point)
                if right_dist < min_dist:
                    min_dist = copy.deepcopy(right_dist)
                    min_point = copy.deepcopy(right_point)
            return min_point
    else:
        right_tree = kdTree['right_set']
        if isinstance(right_tree, dict):
            kdTree['right_set'] = find_similar(target, right_tree)
        else:
            min_dist = np.linalg.norm(target-right_tree)             #计算目标样本与近似点的距离
            min_point = right_tree
            print("直接的最近距离：",min_dist)
            print("直接的最近样本点：",min_point)
            #计算划分平面上的点到目标样本的距离
            if len(kdTree['d_set']) >= 1:
                d_dist = min(np.linalg.norm((target - kdTree['d_set']), axis=1))
                d_idx = np.argmin(np.linalg.norm((target - kdTree['d_set']), axis=1))
                d_point = kdTree['d_set'][d_idx]
                print("分类平面上的最近距离：",d_dist)
                print("分类平面上的最近样本点：",d_point)
                if d_dist < min_dist:
                    min_dist = copy.deepcopy(d_dist)
                    min_point = copy.deepcopy(d_point)
            #计算右边的子集，获取最近点
            if abs(target[kdTree['d']] - kdTree['d_value']) <= min_dist:
                left_tree = kdTree['right_set']
                #左边子树寻找最小点，递归
                print("    新的递归*********")
                left_point = find_minpoint(target, left_tree)
                print("    新的递归结束*********")
                left_dist = np.linalg.norm(target-left_point)
                print("分类另一面的最近距离：",left_dist)
                print("分类另一面的最近样本点：",left_point)
                if left_dist < min_dist:
                    min_dist = copy.deepcopy(left_dist)
                    min_point = copy.deepcopy(left_point)
            return min_point
    return kdTree

#2.2递归向上回退
def find_minpoint(target, kdTree):
    while isinstance(kdTree, dict):
        print("【结构树】:",kdTree)
        new_kdTree = find_similar(target, kdTree)
        kdTree = copy.deepcopy(new_kdTree)
        print("【新结构树】:",kdTree)
        print("====================")
    else:
        print("【树叶节点是】:",kdTree)
        return kdTree

#测试
target = np.array([5.5, -1])
p = find_minpoint(target, kd_Tree)








