# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#数据集
dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
    ]
#特征值列表
labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']
#整理出数据集和标签
X = np.array(dataSet)[:,:8]
Y = np.array(dataSet)[:,8]
#对X进行编码
from sklearn.preprocessing import LabelEncoder
Xdata=np.zeros((17,8))
for i in range(6):
    labelencode = LabelEncoder()
    labelencode.fit(X[:,i])
    Xdata[:,i] = labelencode.transform(X[:,i])
Xdata[:,6:8] = X[:,6:8].astype(float)
#对Y进行编码
labelencode = LabelEncoder()
labelencode.fit(Y)
Ylabel=labelencode.transform(Y)       #得到切分后的数据
labelencode.classes_                        #查看分类标签
labelencode.inverse_transform(Ylabel)    #还原编码前数据


#单隐层BP神经网络
class BPnet(object):
    #1、初始化属性
    def __init__(self):
        self.hidden_N = 0   #隐含层神经元数量
        self.m = 0          #样本数量
        self.n = 0          #属性数量
        self.v = 0          #输入层权重
        self.w = 0          #隐含层权重
        self.dataSet = 0    #数据集X
        self.label = 0      #数据标签Y
        self.error = 1.0e-4 #最小误差
        self.TrueIters = 0  #实际迭代次数
        self.MaxIters = 0   #最大迭代次数
        self.r = 0          #学习率
        self.errList = []   #误差列表
    
    #2、激活函数用对率函数    
    def logit(self, z):
        return 1.0/(1+np.exp(-z))

    #3-1、定义累计误差函数
    def errorfunc(self, Y, Output):
        return sum(0.5*np.power((Output-Y),2))

    #3-2、定义单个样本误差函数
    def errorsample(self, Y, Output):
        return 0.5*np.power((Output-Y),2)
    
    #4、定义对率函数的导数
    def logitD(self, Output):
        return np.multiply(Output, (1-Output))
    
    #5、初始化神经元权重
    def initW(self, row, column):
        return np.random.random((row+1, column))        
    
    #6、训练BP网络
    def train(self, X, Y, hidden_N, r, Iters):
        m, n =np.shape(X)
        v = self.initW(n, hidden_N)         #输入层权重(n+1,h)
        w = self.initW(hidden_N, 1)         #隐含层权重(h+1,1)
        for i in range(Iters):
            ##正向计算输出值
            hi_input = np.hstack((X, np.ones(m).reshape(-1,1)))   #(m,n+1)
            hi_dot = np.dot(hi_input, v)                          #(m,h)
            hi_output = self.logit(hi_dot)
            yi_input = np.hstack((hi_output, np.ones(m).reshape(-1,1)))   #(m,h+1)
            yi_dot = np.dot(yi_input, w)                                  #(m,1)
            yi_output = self.logit(yi_dot)
            ##计算输出结果
            SSE = self.errorfunc(Y, yi_output)
            self.errList.append(SSE[0])
            if SSE<self.error:
                break
            ##反向传播误差，计算隐含层和输入层梯度
            Gra_hidden = np.multiply((yi_output-Y), self.logitD(yi_output))     #(m,1)
            Gra_input = np.multiply(np.dot(Gra_hidden, w[:-1,:].T), self.logitD(hi_output))     #(m,h)
            ##更新权重
            w -= r*np.dot(yi_input.T, Gra_hidden)                    #(h+1,m)x(m,1)
            v -= r*np.dot(hi_input.T, Gra_input)                     #(n+1,m)x(m,h)
        print(w,v,SSE)
        self.m = m
        self.n = n
        self.v = v
        self.w = w
        self.hidden_N = hidden_N
        self.r = r
        self.MaxIters = Iters
        self.TrueIters = i+1
        self.dataSet = X
        self.label = Y

#########数据集测试##############
with open(r"D:\mywork\test\ML\dataSet_BP.txt") as f:
    data = f.readlines()
X = np.array([row.split() for row in data]).astype(float)            
Y = X[:,-1].reshape(-1,1)            
X = X[:,0:-1]
X = (X-X.mean(axis=0))/X.std(axis=0)      
plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], c='b', marker='D')
plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], c='r', marker='o')
plt.show()
    
BP = BPnet()
BP.train(X,Y,4,0.1,500)
err=BP.errList
v = BP.v
w = BP.w
#画等高图
x = np.linspace(-3,3,50)
xx = np.ones((50,50))
xx[:,0:50] = x
yy=xx.T
z=np.ones((50,50))
for i in range(50):
    for j in range(50):
        tempdata = []
        tempdata.append([xx[i,j],yy[i,j],1])    #(1,3)
        tempdata = np.array(tempdata).reshape(1,-1)
        hi = np.dot(tempdata,v)       #隐藏层求点乘积(1,4)
        hi_Output = BP.logit(hi)              #隐藏层输出(1,4)
        yi_Input = np.column_stack((hi_Output,np.ones((1,1))))     #多加一列b构成新的输入项(1,5)
        yi = np.dot(yi_Input,w)     #输出层求点乘积(1,1)
        y_Output = BP.logit(yi)
        z[i,j] = y_Output
for i in range(BP.m):
    if Y[i] == 0:
        plt.scatter(X[i,0],X[i,1],c='b',marker='o')
    else:
        plt.scatter(X[i,0],X[i,1],c='r',marker='^')
plt.contour(x,x,z,1,colors = 'black')
plt.show()

#画误差图
plt.figure()
plt.plot(range(BP.TrueIters),BP.errList,c='r')
plt.show()
##################测试结束######################


##################西瓜3.0进行测试#######################




















