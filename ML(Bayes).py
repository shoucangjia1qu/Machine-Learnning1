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
#     自编Nbayes算法     #
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
class Nbayes(object):
    '''1、设置属性'''
    def __init__(self):
        self.dataSet = 0
        self.labels = 0
        self.docLength = 0
        self.vocabulary = []
        self.vocLength = 0
        self.TF = 0
        self.IDF = 0
        self.TF_IDF = 0
        self.Pcates = {}        #P(y)
        self.tdm = 0            #P(x|Y)

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
            '''3-1生成TF，逐行词加1'''
            for keyword in self.dataSet[row]:
                col = self.vocabulary.index(keyword)
                tf[row,col] += 1
            '''3-2不同句长会有偏差，需要消除'''
            tf[row] = tf[row]/len(self.dataSet[row])
            '''3-3生成IDF，当行有的词就加1'''
            for singleword in set(self.dataSet[row]):
                col = self.vocabulary.index(singleword)
                idf[0,col] += 1
        '''3-4将IDF变成权重'''
        idf = np.log(self.docLength/idf)
        self.TF = tf
        self.IDF = idf
        self.TF_IDF = np.multiply(self.TF,self.IDF)
    
    '''4、计算P(y)'''
    def calPcates(self,labels):
        self.labels = labels
        alllabels = set(labels)
        for temp in alllabels:
            self.Pcates[temp] = labels.count(temp)/len(labels)
    
    '''5、计算P(x|Y)'''
    def calTdm(self):
        self.tdm = np.zeros((len(self.Pcates),self.vocLength))
        sumCatetdm = np.zeros((len(self.Pcates),1))
        for index in range(len(self.TF_IDF)):
            cate = self.labels[index]
            cateList = list(self.Pcates.keys())
            Idx = cateList.index(cate)
            self.tdm[Idx] += self.TF_IDF[index]
            sumCatetdm[Idx] = self.tdm[Idx].sum()
        self.tdm = self.tdm/sumCatetdm
    
    '''6、训练数据'''
    def train(self,data,labels):
        self.processData(data)
        self.calTfidf()
        self.calPcates(labels)
        self.calTdm()
    
    '''7、预测数据'''
    def predict(self,testSet):
        '''7-1将测试集映射到词向量中'''
        testVec = np.zeros((1,self.vocLength))
        for keywrd in testSet:
            if keywrd not in self.vocabulary:
                continue
            col = self.vocabulary.index(keywrd)
            testVec[0,col] += 1
        '''7-2预测测试集'''
        preV = 0
        preC = 0
        for tdm,key in zip(self.tdm,self.Pcates.keys()):
            temp = np.sum(testVec*tdm*self.Pcates[key])
            if temp>preV:
                preV = temp
                preC = key
        return preC,preV
'''正式程序'''
Nb = Nbayes()
Nb.train(dataSet,labels)
'''测试集'''
testSet = []
testlabels = []
testfiles = os.listdir(r"D:\mywork\test\ML_Chinese\Myself1110_testfiles")
for t in testfiles:
    #分词
    text = readfiles("D:\\mywork\\test\\ML_Chinese\\Myself1110_testfiles\\"+t).decode("GBK","ignore")
    text = text.replace("\r\n","").strip()
    text_seg = jieba.cut(text)
    #使用停用词
    segList = [i for i in text_seg if i not in stopwords]
    testSet.append(segList)
    if t[0]=="1":
        testlabels.append(0)        #计算机类文本
    elif t[0]=="4":
        testlabels.append(1)        #交通类文本
    else:
        testlabels.append(2)        #环境类文本

te = []
for i in range(len(testSet)):
    C,V = Nb.predict(testSet[i])
    te.append(C)






  


