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
            break
        else:
            content=readfile(dirname+"\\"+file).strip()   #读取文本内容
            content=content.decode("GBK","ignore")        
            content=content.replace("\r\n","").strip()      #去除换行等
            f=jieba.cut(content)        #开始分词
            savefile(savepath+"\\"+file," ".join(f))      #存储分词后文件
            print(file+'分词成功')        















