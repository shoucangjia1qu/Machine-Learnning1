# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:21:33 2018

@author: ecupl
"""
#####################推荐算法#######################
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import operator
import os

os.chdir(r'D:\mywork\test\command\lastfm-dataset-360K\lastfm-dataset-360K')
curpath = os.getcwd()
#with open("users.txt","rb") as f:
#    users = f.readlines
'''读入数据'''
users = pd.read_csv('usersha1-profile.tsv',sep='\t',header=None)
users = users.rename(columns={0:'id',1:'gender',2:'age',3:'country',4:'data'})
music = pd.read_csv('usersha1-artmbid-artname-plays.tsv',sep='\t',header=None)
music = music.rename(columns={0:'id',1:'artistId',2:'artistName',3:'times'})

######################################
#                                    #
#            冷启动问题               #
#                                    #
######################################
'''一、利用用户注册信息'''
'''1、区分出不同类别的分类变量'''
'''1-1gender'''
users.gender=users.gender.fillna('o')
gender = list(set(users.gender))
'''1-2country'''
countryCount = users.country.value_counts()
countryList = countryCount[countryCount<=5000].index.tolist()
users.country[users['country'].isin(countryList)] = 'other'
country = list(users.country.value_counts().index)
'''1-3age'''
bins = [0,10,20,30,40,50,60,70,80,90,100]
users.age = pd.cut(users.age,bins,labels=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10'])
users.age=users.age.cat.add_categories('a0')
users.age = users.age.fillna('a0')
age = list(users.age.value_counts().index)

'''2、按类别划分不同客户群体'''
userCate=[]
userDict = dict()
n = 0
for i in gender:
    for j in age:
        for k in country:
            userCate.append(i+j+k)
            userDict[n]=users.id[users['gender']==i][users['age']==j][users['country']==k].tolist()
            n += 1
'''3、选取用户的训练集和测试集'''
train = []
test = []
np.random.seed(1234)
for i in users.id:
    if np.random.randint(0,10) == 0:
        test.append(i)
    else:
        train.append(i)
'''4、训练每个用户所属客户群体的艺术家偏好'''
music.artistName = music.artistName.fillna('nothing')
musicList = list(music.artistName.value_counts().index)
userCate_Music = np.zeros((len(musicList),len(userCate)))
music_array = np.array(music)
for x in range(len(music_array)):
    '''4-1确定用户所属群体的Index'''
    if music_array[x,0] in train:
        temp = users[users['id']==U]
        Ugender = temp.gender[0]
        Uage = temp.age[0]
        Ucountry = temp.country[0]
        userCateIdx = userCate.index(Ugender+Uage+Ucountry)
        '''4-1确定艺术家的Index'''
        artistIdx = musicList.index(music_array[x,2])
        '''4-2相应位置+1'''
        userCate_Music[artistIdx,userCateIdx] += 1
    










