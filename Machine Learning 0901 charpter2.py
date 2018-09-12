# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:27:40 2018

@author: ecupl
"""

###################文本挖掘与文本分类##################
'''文本预处理 -> 中文分词 -> 构成词向量空间 -> 权重策略 -> 分类器 -> 分类结果评价'''
import os,jieba,sys
#sys.setdefaultencoding('utf-8')
os.chdir("D:/mywork/test/ML_Chinese")
curpath=os.getcwd()
'''定义读取文件函数'''
def readfile(path):
    with open(path,'rb') as f:
        content = f.read()
    return content
'''定义存储文件函数，以二进制模式保存'''
def savefile(path,content):
    with open(path,'wb') as f:
        f.write(content.encode("GBK"))
'''储存文件，以GBK编码保存'''
def savefile(path,content):
    with open(path,'w') as f:
        f.write(content)

'''进入主程序'''
savedir = "\\分词结果"     #分词结果保存文件夹
subdirs = os.listdir(curpath)       #找到目录下文件
for dirname in subdirs:
    files = os.listdir(curpath+"\\"+dirname)
    savepath = dirname+savedir      #分词结果保存路径
    if not os.path.exists(savepath):    #生成保存文件夹
        os.mkdir(savepath)
    for file in files:
        if file == "分词结果":
            continue
        else:
            content=readfile(dirname+"\\"+file).strip()   #读取文本内容
            content=content.decode("GBK","ignore")        
            content=content.replace("\r\n","").strip()      #去除换行等
            f=jieba.cut(content)        #开始分词
            savefile(savepath+"\\"+file," ".join(f))      #存储分词后文件
            print(file+'分词成功')        

'''转换为Bunch类'''
wordbag_path = "train_set.dat"
from sklearn.datasets.base import Bunch
bunch=Bunch(target_name=[],label=[],filename=[],contents=[])
bunch.target_name.extend(subdirs)
for dirname in subdirs:
    classpath = dirname+savedir
    filenames = os.listdir(classpath)
    for file in filenames:
        fullpath = classpath+"\\"+file
        bunch.label.append(dirname)
        bunch.filename.append(fullpath)
        bunch.contents.append(readfile(fullpath).strip())

'''对象持久化'''
import pickle
with open(curpath+"\\"+wordbag_path,"wb") as obj:
    pickle.dump(bunch,obj)

'''导入停用词'''
stop_path = curpath+"\\stop\\stopwords.txt"
stoplist = readfile(stop_path).splitlines()

###################将分词变为词向量并储存##################
'''TF-IDF方法'''
'''
TF:文本在向量中出现的概率分布，词频
IDF:文本在词袋中出现的频率
权重策略：词频高，且在其他文章中很少出现，适合用来分类
'''
import sys,os,pickle    #引入持久化类
from imp import reload
from sklearn.datasets.base import Bunch    #引入Bunch类
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer    #TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer     #TF-IDF向量生成类
'''配置输出UTF-8'''
reload(sys)
sys.setdefaultencoding("utf-8")
'''读取Bunch对象'''
def readbunchobj(path):
    with open(path,"rb") as file_obj:
        bunch = pickle.load(file_obj)
        return bunch
'''写入bunch对象'''
def writebunchobj(path,bunch):
    with open(path,"wb") as obj:
        pickle.dump(bunch,obj)
'''主程序'''
'''1、导入分词后的词向量Bunch对象'''
bunch_path = curpath+"\\train_set.dat"
bunch = readbunchobj(bunch_path)
'''2、构建TF-IDF词向量空间对象'''
tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filename,
                   tdm=[], vocabulary={})
'''3、使用TfidfVectorizer初始化向量空间模型'''
vectorizer = TfidfVectorizer(stop_words=stoplist, sublinear_tf=True, max_df=0.5)
transformer = TfidfTransformer()    #统计每个词语的TF-IDF权重
text=[i.decode("GBK","ignore") for i in bunch.contents]     #将二进制转为unicode
tfidfspace.tdm = vectorizer.fit_transform(text)
tfidfspace.vocabulary = vectorizer.vocabulary_
'''4、持久化向量词袋'''
space_path = curpath+"\\tfidfspace.dat"
writebunchobj(space_path,tfidfspace)


#########################################
#                                       #
#               实例                    #
#                                       #
#########################################
'''1、读入训练集bunch对象'''
import pickle
train_path = curpath+"\\tfidfspace.dat"
with open(curpath+"\\tfidfspace.dat","rb") as file:
    train_bunch = pickle.load(file)
'''2、测试集'''
import os,jieba
'''二进制读入文件'''
def readf(path):
    with open(path,'rb') as f:
        content = f.read()
        return content
'''二进制写入文件'''
def savef(path,content):
    with open(path,'wb') as f:
        f.write(content.encode("GBK"))
test_path = curpath+"\\test"
'''遍历测试文件夹下文件'''
test_dir = os.listdir(test_path)
test_savedir = test_path+"\\result"
for i in test_dir:
    if i == "result":
        continue
    else:
        test_filepath = test_path+"\\"+i
        if not os.path.exists(test_savedir):
                os.mkdir(test_savedir)
        for j in os.listdir(test_filepath):
            test_filename = test_filepath+"\\"+j
            content=readf(test_filename)
            content=content.decode("GBK","ignore")        
            content=content.replace("\r\n","").strip()      #去除换行等
            cutwds=jieba.cut(content)        #开始分词
            savef(test_savedir+"\\"+j," ".join(cutwds))      #存储分词后文件
            print(j+'分词成功') 
'''构建测试集Bunch类'''
testbag_path = "test\\test_set.dat"
from sklearn.datasets.base import Bunch
testbunch = Bunch(target_name=[], label=[], filenames=[],contents=[])
testbunch.target_name.extend(test_dir)
for subdir in test_dir:
    if subdir=="result":
        continue
    tfilepath = test_path+"\\"+subdir
    filenames = os.listdir(tfilepath)
    for file in filenames:
        fullpath = test_savedir+"\\"+file
        testbunch.label.append(subdir)
        testbunch.filenames.append(fullpath)
        testbunch.contents.append(readfile(fullpath).strip())
'''持久化'''
with open(curpath+"\\"+testbag_path,"wb") as obj:
    pickle.dump(testbunch,obj)
'''写入测试集数据'''
with open(curpath+"\\"+testbag_path,'rb') as testf:
    testbunch = pickle.load(testf)
'''导入停用词'''
stop_path = curpath+"\\stop\\stopwords.txt"
stoplist = readfile(stop_path).splitlines()
'''构建词向量对象'''
tfidftest = Bunch(target_name=testbunch.target_name, label=testbunch.label, filenames=testbunch.filenames,
                   tdm=[], vocabulary={})
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer    #TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer     #TF-IDF向量生成类
'''构建测试集向量时需使用训练集词袋向量'''
vectorizer = TfidfVectorizer(stop_words=stoplist, sublinear_tf=True, max_df=0.5,
                             vocabulary=train_bunch.vocabulary)
transformer = TfidfTransformer()    #统计每个词语的TF-IDF权重
text=[i.decode("GBK","ignore") for i in testbunch.contents]     #将二进制转为unicode
tfidftest.tdm = vectorizer.fit_transform(text)
tfidftest.vocabulary = train_bunch.vocabulary
'''3、使用贝叶斯分类对文本进行分类'''
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.001).fit(train_bunch.tdm,train_bunch.label)
'''4、预测'''
pre=clf.predict(tfidftest.tdm)
pro=clf.predict_proba(tfidftest.tdm)
'''5、查看精度'''
la=tfidftest.label
file=tfidftest.filenames
n=0
for a,b,c in zip(la,file,pre):
    print(a,b,c)
    if a==c:
        n+=1
auc=n/len(pre)
print("AUC:%.3f"%auc)
'''6、评估'''
import numpy as np
import pandas as pd
from sklearn import metrics
mt=metrics.classification_report(la,pre)


#########################################
#                                       #
#               推导                    #
#实现朴素贝叶斯算法                      #
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
        self.testset=0          #训练集
'''3、导入和训练数据集'''
def train_set(self,trainset,classVec):
    self.cate_prob(classVec)        #计算每个分类的概率，函数
    self.doclength=len(trainset)    #文本长度
    tempset=set()                   #生成字典key值的集合
    [tempset.add(word) for doc in trainset for word in doc]     #不重复地合成每个分词
    self.vocabulary = list(tempset) #转换为词典
    self.vocablen = len(self.vocabulary)      #词典词长
    self.wordfre(trainset)          #统计词频数据集，函数
    self.buildtdm()                 #计算P(x|y)条件概率，函数
'''4、计算P(y)的概率'''
def cate_prob(self,classVec):
    self.labels=classVec            #获取所有的分类数据
    labeltemps=set(self.labels)     #获取全部分类类别组成的集合
    for labeltemp in labeltemps:
        classtimes=self.labels.count(labeltemp)     #统计某个类别的频次
        self.Pcates[labeltemp]=float(classtimes)/float(len(self.labels))    #计算某个类别出现的概率
'''5、'''
        
    
    


