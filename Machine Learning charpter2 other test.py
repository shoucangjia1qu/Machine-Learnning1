# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:27:40 2018

@author: ecupl
"""
'''导入简单数据'''
def loadDataSet():
    postingList=['my dog has flea problems help please',
                 'maybe not take him to dog park stupid',
                 'my dalmation is so cute I love him my',
                 'stop posting stupid worthless garbage',
                 'mr licks ate my steak how to stop him',
                 'quit buying worthless dog food stupid']
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
train,classv=loadDataSet()
'''生成文本向量矩阵'''
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer    #TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer     #TF-IDF向量生成类
vectorizer = TfidfVectorizer( sublinear_tf=True)
transformer = TfidfTransformer()    #统计每个词语的TF-IDF权重
tdm = vectorizer.fit_transform(train)
tvocabulary = vectorizer.vocabulary_

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             vocabulary=tvocabulary)
transformer = TfidfTransformer()    #统计每个词语的TF-IDF权重
testtdm = vectorizer.fit_transform(train)
testvocabulary = tvocabulary
'''贝叶斯模型训练'''
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.001).fit(tdm,classv)
'''预测'''
pre=clf.predict(testtdm)
pro=clf.predict_proba(testtdm)
'''查看精度'''
la=classv
n=0
for a,b in zip(la,pre):
    print(a,b)
    if a==b:
        n+=1
auc=n/len(pre)
print("AUC:%.3f"%auc)
'''6、评估'''
import numpy as np
import pandas as pd
from sklearn import metrics
mt=metrics.classification_report(la,pre)
print(mt)

#########################################
#                                       #
#               推导                    #
#算法改进                               # 
#########################################
'''1、定义训练集文本、类别标签'''
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him','my'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
'''2、编写贝叶斯算法类，并创建默认的构造方法'''
class NBayes(object):
    def __init__(self):
        self.vocabulary=[]      #词典
        self.idf=0              #词典的IDF权值向量
        self.tf=0               #训练集的权值矩阵
        self.tdm=0              #P(x|y)
        self.Pcates={}          #P(y)类别字典
        self.labels=[]          #每个文本分类
        self.doclength=0        #训练集本文长度
        self.vocablen=0         #词典词长
        self.testset=0          #测试集
    '''3、导入和训练数据集'''
    def train_set(self,trainset,classVec):
        self.cate_prob(classVec)        #计算每个分类的概率，函数
        self.doclength=len(trainset)    #训练集文本数
        tempset=set()                   #生成字典key值的集合
        [tempset.add(word) for doc in trainset for word in doc]     #不重复地合成每个分词
        self.vocabulary = list(tempset) #转换为词典
        self.vocablen = len(self.vocabulary)      #词典词长
        self.wrd_tfidf(trainset)          #统计词频数据集，函数
        self.build_tdm()                 #计算P(x|y)条件概率，函数
    '''4、计算P(y)的概率'''
    def cate_prob(self,classVec):
        self.labels=classVec            #获取所有的分类数据
        labeltemps=set(self.labels)     #获取全部分类类别组成的集合
        for labeltemp in labeltemps:
            classtimes=self.labels.count(labeltemp)     #统计某个类别的频次
            self.Pcates[labeltemp]=float(classtimes)/float(len(self.labels))    #计算某个类别出现的概率
    '''5、生成tf-idf'''
    def wrd_tfidf(self,trainset):
        self.idf = np.zeros([1,self.vocablen])      #1x词典数的矩阵
        self.tf = np.zeros([self.doclength,self.vocablen])      #训练集文本数x词典数的矩阵
        for index in range(self.doclength):
            for word in trainset[index]:
                self.tf[index,self.vocabulary.index(word)] += 1
            '''消除不同句长导致的偏差'''
            self.tf[index] = self.tf[index]/float(len(trainset[index]))
            for singlewrd in set(trainset[index]):
                self.idf[0,self.vocabulary.index(singlewrd)] += 1
        '''idf变成权重'''
        self.idf = np.log(float(self.doclength)/self.idf)
        '''tf变成权重下的tf-idf'''
        self.tf = np.multiply(self.tf,self.idf)
        
    '''6、生成每维值P(x|y)矩阵'''
    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates),self.vocablen])       #每个类别的词频统计
        sumlist = np.zeros([len(self.Pcates),1])        #每个类别的词频汇总
        for index in range(self.doclength):
            self.tdm[self.labels[index]] += self.tf[index]
            sumlist[self.labels[index]] = np.sum(self.tdm[self.labels[index]])
        self.tdm = self.tdm/sumlist         #生成P(x|y)最终矩阵
    '''7、将测试集映射到当前词典中'''
    def map2vocab(self,testdata):
        self.testset = np.zeros([1,self.vocablen])
        for word in testdata:
            self.testset[0,self.vocabulary.index(word)]+=1
    '''8、预测函数'''   
    def predict(self,testset):
        if np.shape(testset)[1] != self.vocablen:
            print("输入错误")
            exit(0)
        else:
            predvalue=0     #初始化类别概率
            predclass=""    #初始化类别名臣
            for tdm_vect,keyclass in zip(self.tdm,self.Pcates):
                #计算P(x|y)*P(y)
                temp = np.sum(testset*tdm_vect*self.Pcates[keyclass])
                if temp > predvalue:
                    predvalue = temp
                    predclass = keyclass
            return predclass
'''查看结果'''
dataset,listclass = loadDataSet()
nb = NBayes()       #实例化
nb.train_set(dataset,listclass)
nb.map2vocab(dataset[1])
print(nb.predict(nb.testset))


#########################################
#                                       #
#        分类算法：KNN                  #
#                                      # 
#########################################
import os
import numpy as np
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
dataset,label = createDataSet()
'''绘图'''
fig=plt.figure()
ax = fig.add_subplot(111)
index=0
'''加入点'''
for point in dataset:
    if label[index]=="A":
        ax.scatter(point[0],point[1],c='blue',marker='o',linewidths=0,s=300)
        plt.annotate("("+str(point[0])+","+str(point[1])+")", xy=(point[0],point[1]))
    else:
        ax.scatter(point[0],point[1],c='red',marker='^',linewidths=0,s=300)
        plt.annotate("("+str(point[0])+","+str(point[1])+")", xy=(point[0],point[1]))
    index+=1
'''新加入一个点'''
testdata=[0.2,0.2]
ax.scatter(testdata[0],testdata[1],c='green',marker='^',linewidths=0,s=300)
plt.annotate("("+str(testdata[0])+","+str(testdata[1])+")", xy=(testdata[0],testdata[1]))
'''展示图'''
plt.show()

'''1、导入库'''
import os
import numpy as np
import operator
nb = NBayes()
'''2、构造夹角余弦公式'''
def cosdist(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
'''3、KNN实现分类器'''
def classify(testdata,trainset,listclass,k):
    datasetsize = trainset.shape[0]     #训练集矩阵行数
    distances = np.zeros(datasetsize)
    for index in range(datasetsize):
        distances[index]=cosdist(testdata,trainset[index])      #计算向量间的距离
    sortdist = np.argsort(-distances)   #倒序排列索引，因为夹角余弦越大越接近
    classcount={}
    for i in range(k):
        votelabel = listclass[sortdist[i]]
        classcount[votelabel]=classcount.get(votelabel,0) + 1   #获取标签，没有的标签默认为0
    '''对分类字典按照value值重新排序'''
    sortclasscount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)     #根据第2个阈值来降序排列
    return sortclasscount[0][0]
'''4、评估分类结果'''
dataset,listclass=loadDataSet()
nb=NBayes()
nb.train_set(dataset,listclass)
print(classify(nb.tf[3],nb.tf,listclass,3))
'''准确率达100%'''








