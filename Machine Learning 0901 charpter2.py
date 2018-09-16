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
        self.doclength=len(trainset)    #训练集文本数
        tempset=set()                   #生成字典key值的集合
        [tempset.add(word) for doc in trainset for word in doc]     #不重复地合成每个分词
        self.vocabulary = list(tempset) #转换为词典
        self.vocablen = len(self.vocabulary)      #词典词长
        self.wrd_freq(trainset)          #统计词频数据集，函数
        self.build_tdm()                 #计算P(x|y)条件概率，函数
    '''4、计算P(y)的概率'''
    def cate_prob(self,classVec):
        self.labels=classVec            #获取所有的分类数据
        labeltemps=set(self.labels)     #获取全部分类类别组成的集合
        for labeltemp in labeltemps:
            classtimes=self.labels.count(labeltemp)     #统计某个类别的频次
            self.Pcates[labeltemp]=float(classtimes)/float(len(self.labels))    #计算某个类别出现的概率
    '''5、生成普通词频向量'''
    def wrd_freq(self,trainset):
        self.idf = np.zeros([1,self.vocablen])      #1x词典数的矩阵
        self.tf = np.zeros([self.doclength,self.vocablen])      #训练集文本数x词典数的矩阵
        for index in range(self.doclength):
            for word in trainset[index]:
                self.tf[index,self.vocabulary.index(word)] += 1
            for singlewrd in set(trainset[index]):
                self.idf[0,self.vocabulary.index(singlewrd)] += 1
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
            predclass=""    #初始化类别名称
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
nb.map2vocab(dataset[0])
print(nb.predict(nb.testset))


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
        self.testset=0          #训练集
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








