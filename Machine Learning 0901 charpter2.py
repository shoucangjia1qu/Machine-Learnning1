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

###################训练模型##################
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

            










