# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:04:34 2018

@author: ecupl
"""

###################逐次逼近法/迭代法求解##################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''设计BP网络，含1个隐含层'''
class BPNet(object):
    '''1、定义属性'''
    def __init__(self):
        #人工定的参数
        self.eb = 0.01      #误差容限
        self.r = 0.1       #学习率
        self.mc = 0.3       #栋梁因子，用以考虑上次迭代的权重的结果
        self.max_iterator = 2000    #最大迭代次数
        self.nHidden = 4    #隐含层神经元个数
        self.nOutput = 1    #输出层输出个数
        #系统迭代生成的参数
        self.iterator = 0   #迭代次数
        self.errorList = [] #每次迭代的误差列表
        self.dataSet = 0    #训练集数据
        self.Labels = 0     #训练集分类标签
        self.rows = 0       #训练集行数
        self.cols = 0       #训练集列数
        self.hiddenWB = 0
        self.outputWB = 0
        self.Y = 0          #输出标签
    '''2、定义误差函数'''
    def errorfunc(self,singleError):
        return(np.sum(np.power(singleError,2))*0.5)       #0.5*Sigma((Y-O)**2)
    '''3、定义激活函数'''
    def logit(self,net):
        return(1.0/(1.0+np.exp(-net)))
    '''4、定义传递函数导函数'''
    def dlogit(self,y):
        return(np.multiply(y,(1.0-y)))
    '''5、初始化隐含层权重(-1,1)'''
    def init_hiddenWB(self):
        self.hiddenWB = 2*(np.random.rand(self.nHidden,self.cols+1)-0.5)    #(4,3)
    '''6、初始化输出层权重(-1,1)'''
    def init_outputWB(self):
        self.outputWB = 2*(np.random.rand(self.nOutput,self.nHidden+1)-0.5) #(1,5)
    '''7、加载数据集'''
    def loadData(self,path):
        with open(path,"r") as f:
            content = f.readlines()
        tempList = [row.split() for row in content]
        m,n = np.shape(tempList)
        data = np.zeros((m,n-1))
        label = np.zeros((m,1))
        for i in range(m):
            for j in range(n):
                if j != n-1:
                    data[i,j] = tempList[i][j]
                else:
                    label[i,0] = tempList[i][j]
        self.dataSet = data
        self.Labels = label
        self.rows = m
        self.cols = n-1        
    '''8、数据归一化/标准化'''
    def normalize(self,dataSet):
        m,n = np.shape(dataSet)
        for i in range(n):
            dataSet[:,i] = (dataSet[:,i]-np.mean(dataSet[:,i]))/np.std(dataSet[:,i]+1.0e-10)
        self.dataSet = dataSet
    '''9、主函数'''
    def BPtrain(self):
        data = self.dataSet
        Y = self.Labels
        self.init_hiddenWB()
        self.init_outputWB()
        hiddenWBold = outWBold = 0      #设置前一次隐含层和输出层权重为0
        data = np.column_stack((data,np.ones((self.rows,1))))
        for i in range(self.max_iterator):
            hi = np.dot(self.hiddenWB,data.T)       #隐藏层求点乘积(4,307)
            hi_Output = self.logit(hi)              #隐藏层输出(4,307)
            yi_Input = np.row_stack((hi_Output,np.ones((1,self.rows))))     #多加一列b构成新的输入项(5,307)
            yi = np.dot(self.outputWB,yi_Input)     #输出层求点乘积(1,307)
            y_Output = self.logit(yi)               #输出层输出(1,307)
            '''反向传播过程，计算误差'''
            err = Y.T - y_Output        #每个样本的误差(1,307)
            sse = self.errorfunc(err)   #计算总体误差
            self.errorList.append(sse)  #记录当前总体误差
            #停止主循环条件
            if sse<=self.eb:        
                self.iterator = i+1
                break
            #计算梯度
            deltaO = np.multiply(err,self.dlogit(y_Output))  #输出层梯度(1,307)
            deltaH = np.multiply(np.dot(self.outputWB[:,:-1].T,deltaO),self.dlogit(hi_Output))  #隐含层梯度(4,307)
            #更新权重
            if i==0:
                self.outputWB = self.outputWB + self.r*np.dot(deltaO,yi_Input.T)
                self.hiddenWB = self.hiddenWB + self.r*np.dot(deltaH,data)
            else:
                self.outputWB = self.outputWB + (1-self.mc)*self.r*np.dot(deltaO,yi_Input.T) + self.mc*outWBold
                self.hiddenWB = self.hiddenWB + (1-self.mc)*self.r*np.dot(deltaH,data) + self.mc*hiddenWBold
            outWBold = np.dot(deltaO,yi_Input.T)
            hiddenWBold = np.dot(deltaH,data)
            self.Y = y_Output

#正式程序
for now in range(100):
    bp = BPNet()       
    bp.loadData("D:\\mywork\\test\\ML\\dataSet_BP.txt")
    bp.normalize(bp.dataSet)            
    bp.BPtrain()          
    print(bp.errorList[-1])
    if bp.errorList[-1]<=1:
        break

    #隐含层和输出层权重
hw = bp.hiddenWB
ow = bp.outputWB
#画散点图
data = bp.dataSet
labels = bp.Labels
plt.figure()
for i in range(bp.rows):
    if labels[i] == 0:
        plt.scatter(data[i,0],data[i,1],c='b',marker='o')
    else:
        plt.scatter(data[i,0],data[i,1],c='r',marker='^')
plt.show()
#准备画等高图
x = np.linspace(-3,3,50)
xx = np.ones((50,50))
xx[:,0:50] = x
yy=xx.T
z=np.ones((50,50))
for i in range(50):
    for j in range(50):
        tempdata = []
        tempdata.append([xx[i,j],yy[i,j],1])    #(1,3)
        tempdata = np.array(tempdata)
        hi = np.dot(hw,tempdata.T)       #隐藏层求点乘积(4,1)
        hi_Output = bp.logit(hi)              #隐藏层输出(4,1)
        yi_Input = np.row_stack((hi_Output,np.ones((1,1))))     #多加一列b构成新的输入项(5,1)
        yi = np.dot(ow,yi_Input)     #输出层求点乘积(1,1)
        y_Output = bp.logit(yi)
        z[i,j] = y_Output
plt.figure()
for i in range(bp.rows):
    if labels[i] == 0:
        plt.scatter(data[i,0],data[i,1],c='b',marker='o')
    else:
        plt.scatter(data[i,0],data[i,1],c='r',marker='^')
plt.contour(x,x,z,1,colors = 'black')
plt.show()

#画误差图
plt.figure()
plt.plot(range(bp.max_iterator),bp.errorList,c='r')
plt.show()
