# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:21:33 2018

@author: ecupl
"""
#####################推荐算法#######################
import numpy as np
from numpy import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

os.chdir(r'D:\mywork\test\command')
curpath = os.getcwd()
#with open("users.txt","rb") as f:
#    users = f.readlines
'''读入数据'''
users = pd.read_table('users.dat',sep = '::',header=None,engine='python')
movies = pd.read_table('movies.dat',sep = '::',header=None,engine='python')
ratings = pd.read_table('ratings.dat',sep = '::',header=None,engine='python')
'''修改列名'''
users.columns=['UserID','Gender','Age','Occupation','Zip-code']
movies.rename(columns={0:'MovieID',1:'Title',2:'Genres'}, inplace = True)
ratings.rename(columns={0:'UserID',1:'MovieID',2:'Rating',3:'Timestamp'}, inplace=True)

'''切分数据集,交叉验证啊'''
def splitData(data, M, k,seed):
    test = []
    train = []
    random.seed(seed)
    for user, item in data:
        if random.randint(0,M) == k:
            test.append([user,item])
        else:
            train.append([user,item])
    return train,test



