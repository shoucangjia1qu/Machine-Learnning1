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
vectorizer = TfidfVectorizer()
transformer = TfidfTransformer()    #统计每个词语的TF-IDF权重
tdm = vectorizer.fit_transform(train)
tvocabulary = vectorizer.vocabulary_

vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.5,
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

#########################################
#                                       #
#        特征提取                       #
#                                      # 
#########################################
'''1、字典特征提取'''
'''分类特征是“属性值”对，其中该值被限制为
不排序的可能性的离散列表（例如，主题标识符，对象类型，标签，名称...）'''
from sklearn.feature_extraction import DictVectorizer
#分类属性+数字特征的提取
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},
 ]
vec = DictVectorizer()      #模型
fea=vec.fit_transform(measurements)     #训练
fea.toarray()
vec.vocabulary_             #输出字典
vec.get_feature_names()     #输出列表
#转换为稀疏二维矩阵，相当于get_dummies
measurements = [
    {'city': 'Dubai', 'temperature': 'high'},
    {'city': 'Dubai', 'temperature': 'low'},
    {'city': 'Dubai', 'temperature': 'low', 'geo':'east'}

 ]
vec = DictVectorizer()      #模型
fea=vec.fit_transform(measurements)     #训练
fea.toarray()
vec.vocabulary_             #输出字典
vec.get_feature_names()     #输出列表



'''2、文本特征提取'''
'''2-1统计词频和正则化切词，生成TF'''
from sklearn.feature_extraction.text import CountVectorizer
data=['my dog has flea problems help please',
                 'maybe not take him to dog park stupid',
                 'my dalmation is so cute I love him my',
                 'stop posting stupid worthless garbage',
                 'mr licks ate my steak how to stop him',
                 'quit buying worthless dog food stupid']
vec = CountVectorizer(min_df=1)
x=vec.fit_transform(data)
x.toarray()
vec.vocabulary_             #输出字典
vec.get_feature_names()     #输出列表
#切分句子，使用默认区分块。token_pattern的正则
'''
CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)
'''
analyze=CountVectorizer(min_df=1).build_analyzer()
analyze('你好,下午好!今天天气不错.')
#['你好', '下午好', '今天天气不错']
'''自定义矢量化器类'''
def my_tokenizer(s):
    return s.split()
vectorizer = CountVectorizer(tokenizer=my_tokenizer)
vectorizer.build_analyzer()(u"Some... punctuation!")
#['some...', 'punctuation!']

'''2-2TF-IDF权重计算'''
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)        #非默认参数
'''
TfidfTransformer(norm='l2', smooth_idf=False, sublinear_tf=False,
         use_idf=True)
'''
count=x.toarray()
tfidf=transformer.fit_transform(count)      #L2范数标准化
tfidf.toarray()

'''2-3结合TF，TF-IDF功能的模型'''
from sklearn.feature_extraction.text import TfidfVectorizer     #TF-IDF向量生成类
vec=TfidfVectorizer()
x=vec.fit_transform(data)
x.toarray()
vec.vocabulary_             #输出字典
vec.get_feature_names()     #输出列表


'''3、hash值映射'''
from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=10)       #哈希函数-1到1，但是解释性不强
hv.fit_transform(data).toarray()
'''
    一般来说，只要词汇表的特征不至于太大，大到内存不够用，肯定是使用一般意义的向量化比较好。
因为向量化的方法解释性很强，我们知道每一维特征对应哪一个词，进而我们还可以使用TF-IDF对
各个词特征的权重修改，进一步完善特征的表示。
    而Hash Trick用大规模机器学习上，此时我们的词汇量极大，使用向量化方法内存不够用，而使用
Hash Trick降维速度很快，降维后的特征仍然可以帮我们完成后续的分类和聚类工作。当然由于分布
式计算框架的存在，其实一般我们不会出现内存不够的情况。因此，实际工作中我使用的都是特征向量化。
'''

def hashing_vectorizer(features, N):
     x = N * [0]
     for f in features:
         h = hash(f)
         idx = h % N
         x[idx] += 1
     return x
 
 hashing_vectorizer(["cat","dog","cat"],4)   
    
    
    
if f == 'cat':

    hash(f) = 1

elif f == 'dog':

    hash(f) = 2
