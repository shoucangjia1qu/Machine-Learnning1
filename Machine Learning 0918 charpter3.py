# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:27:40 2018

@author: ecupl
"""

###################决策树##################
'''1、了解算法ID3——信息增益'''
import numpy as np
'''计算分类别信息熵'''
b = 128+60+64+64+64+132+64+32+32      #购买
nb = 64+64+64+128+64                  #未购买
E_cate = -((b/(b+nb))*np.log2(b/(b+nb)) + (nb/(b+nb))*np.log2(nb/(b+nb)))   #分类信息熵
'''计算子节点信息熵'''
y = 64+64+128+64+64     #年轻类
yb = 64+64                  #年轻购买
ynb = 64+64+128             #年轻不购买
E_y = -((yb/(yb+ynb))*np.log2(yb/(yb+ynb)) + (ynb/(yb+ynb))*np.log2(ynb/(yb+ynb)))
e = 128+64+32+32        #中年类
eb = 128+64+32+32           #中年购买
E_e = -(eb/(eb+enb))*np.log2(eb/(eb+enb))
o = 60+64+64+132+64     #老年类
ob = 60+64+133              #年轻购买
onb = 64+63                 #年轻不购买
E_o = -((ob/(ob+onb))*np.log2(ob/(ob+onb)) + (onb/(ob+onb))*np.log2(onb/(ob+onb)))
'''计算信息增益'''
Py = y/(y+e+o)
Pe = e/(y+e+o)
Po = o/(y+e+o)
G_age = E_cate - (Py*E_y + Pe*E_e + Po*E_o)
'''G_age=0.2666969558634843'''

'''ID3算法实现'''
import numpy as np
import math, copy, pickle

class ID3Tree(object):
    '''1、初始化'''
    def __init__(self):         #构造方法
        self.tree={}            #生成树
        self.dataset=[]         #数据集
        self.labels=[]          #标签集
        
    '''2、数据导入函数'''
    def loadDataSet(self,path,labels):
        datalist=[]
        with open(path,"rb") as f:          #二进制形式读取文件
            content=f.read()
            rows = content.splitlines()     #分割文本，按行转换为一维表
            datalist=[row.split("\t") for row in rows if row.strip()]  #用制表符分割每个样本的变量值
            self.dataset = datalist
            self.labels = labels
    '''3、执行决策树函数'''
    def train(self):
        labels = copy.deepcopy(self.labels)     #深度复制lebels，相当于备份
        self.tree = self.buildTree(self,dataset,labels)















