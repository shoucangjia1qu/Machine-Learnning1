# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:23:21 2018

@author: ecupl
"""
import numpy as np
import os, jieba
os.chdir(r"D:\mywork\test\ML_Chinese\train")
'''读取文件'''
curpath = os.getcwd()
traffic = os.listdir(curpath)[0]
documents = os.listdir(traffic)
def readfiles(path):
    with open(path,"rb") as f:
        content = f.read()
    return content
'''分词'''
text0 = readfiles(traffic+'\\'+documents[0]).decode("GBK")
text0 = text0.replace("\r\n","").strip()
text_seg = jieba.cut(text0)
'''加载停用词'''
stopwords = readfiles("D:\\mywork\\test\\ML_Chinese\\stop\\stopwords.txt").decode("GBK","ignore")
stopwords = stopword.strip().split()
stopwords.extend(["【","】","，","。","（","）"," "])
'''完成分词转成列表或者储存'''
segList = [i for i in text_seg if i not in stopwords]
segTxt = ""
for i in text_seg:
    if i not in stopwords:
        segTxt += i
        segTxt += " "
with open("traffic0.txt","w") as f:
    f.write(segTxt)

#########################
#                       #
#     自编TF-IDF算法     #
#                       #
#########################
import jieba, os

'''导入数据并创建词向量。0：计算机，1：交通，2：环境'''
def readfiles(path):
    with open(path,"rb") as f:
        content = f.read()
    return content
path = "D:\\mywork\\test\\ML_Chinese\\Myself1110"
os.chdir(path)
dots = os.listdir(path)
#设置分词停用词
stopwords = readfiles("D:\\mywork\\test\\ML_Chinese\\stop\\stopwords.txt").decode("GBK","ignore")
stopwords = stopwords.strip().split()
stopwords.extend(["【","】","，","。","（","）"," "])
#正式导数和分词
dataSet = []
labels = []
for t in dots:
    #分词
    text = readfiles(t).decode("GBK","ignore")
    text = text.replace("\r\n","").strip()
    text_seg = jieba.cut(text)
    #使用停用词
    segList = [i for i in text_seg if i not in stopwords]
    dataSet.append(segList)
    if t[0]=="1":
        labels.append(0)        #计算机类文本
    elif t[0]=="4":
        labels.append(1)        #交通类文本
    else:
        labels.append(2)        #环境类文本

'''根据现有词向量创建TF-IDF权重'''
class TFIDF(object):
    '''1、设置属性'''
    def __init__(self):
        self.dataSet = 0
        self.docLength = 0
        self.vocabulary = []
        self.vocLength = 0
        self.TF = 0
        self.IDF = 0
        self.TF_IDF = 0

    '''2、处理数据，得到词向量'''
    def processData(self,data):
        self.dataSet = data
        self.docLength = len(data)
        docVocabulary = set()
        for doc in data:
            for keyword in doc:
                docVocabulary.add(keyword)
        self.vocabulary = list(docVocabulary)
        self.vocLength = len(self.vocabulary)

    '''3、计算TF-IDF'''
    def calTfidf(self):
        tf = np.zeros((self.docLength,self.vocLength))
        idf = np.zeros((1,self.vocLength))
        for row in range(self.docLength):
            '''生成TF，逐行词加1'''
            for keyword in self.dataSet[row]:
                col = self.vocabulary.index(keyword)
                tf[row,col] += 1
            '''不同句长会有偏差，需要消除'''
            tf[row] = tf[row]/len(self.dataSet[row])
            '''生成IDF，当行有的词就加1'''
            for singleword in set(self.dataSet[row]):
                col = self.vocabulary.index(singleword)
                idf[0,col] += 1
        '''将IDF变成权重'''
        idf = np.log(self.docLength/idf)
        self.TF = tf
        self.IDF = idf
        self.TF_IDF = np.multiply(self.TF,self.IDF)

tfidf = TFIDF()
tfidf.processData(postingList)
tfidf.calTfidf()







  


