# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:27:40 2018

@author: ecupl
"""

###################文本挖掘与文本分类##################
'''文本预处理 -> 中文分词 -> 构成词向量空间 -> 权重策略 -> 分类器 -> 分类结果评价'''
import os,jieba
os.chdir("D:/mywork/test/ML_Chinese")
curpath=os.getcwd()
'''定义读取文件函数'''
def readfile(path):
    with open(path,'rb') as f:
        content = f.read()
'''定义存储文件函数'''
def savefile(path,content):
    with open(path,'wb') as f:
        f.write(content)
'''进入主程序'''










