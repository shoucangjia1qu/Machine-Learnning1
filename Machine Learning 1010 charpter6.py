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

###################SOM网络##################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''定义SOM算法类'''
class Kohonen(object):
    '''1、定义属性'''
    def __init__(self):
        self.maxRate = 0.8          #最大学习率
        self.minRate = 0.05         #最小学习率
        self.maxRound = 5           #最大聚类半径
        self.minRound = 0.5         #最小聚类半径
        self.steps = 1000           #迭代次数
        self.RateList = []          #学习率列表
        self.RoundList = []         #聚类半径列表
        self.w = []                 #权重
        self.M = 2                  #输出层节点数。MxN表示聚类数，这里是展示为2*2的二维模式
        self.N = 2                 
        self.dataSet = 0            #训练集
        self.Labels = 0             #自身聚类标签
        self.Y = 0                  #聚类结果
    '''2、读入数据集'''
    def loadData(self,path):
        with open(path,"r") as f:
            content = f.readlines()
        tempList = [row.split() for row in content]
        m,n = np.shape(tempList)
        data = np.zeros((m,n-1))
        label = np.zeros((m,1))
        for i in range(m):
            for j in range(n):
                if j != 0:
                    data[i,j-1] = tempList[i][j]
                else:
                    label[i,0] = tempList[i][j]
        self.dataSet = data
        self.Labels = label
    '''3、数据归一化/标准化'''
    def normalize(self,dataSet):
        m,n = np.shape(dataSet)
        for i in range(n):
            dataSet[:,i] = (dataSet[:,i]-np.mean(dataSet[:,i]))/np.std(dataSet[:,i]+1.0e-10)
        return dataSet
    '''4、定义欧氏距离公式'''
    def edist(self,v1,v2):
        return(np.linalg.norm(v1-v2))
    '''5、初始化输出层/竞争层'''
    def out_grid(self):
        grid = np.zeros((self.M*self.N,2))      #分成四类，两个维度
        k = 0
        for i in range(self.M):
            for j in range(self.N):
                grid[k,:] = np.array([i,j])
                k += 1
        return grid
    '''6、学习率和半径'''
    def ratecalc(self,i):                                                          #学习率和半径
    	Learn_rate = self.maxRate-((i+1.0)*(self.maxRate-self.minRate))/self.steps
    	R_rate     = self.maxRound-((i+1.0)*(self.maxRound-self.minRound))/self.steps
    	return  Learn_rate,R_rate

    '''6、主程序'''
    def train(self):
        m,n = self.dataSet.shape
        normData = self.normalize(self.dataSet)         #数据归一化
        grid = self.out_grid()                          #输出层初始化
        self.w = np.random.rand(n,self.M*self.N)        #随机初始化权重向量
        if self.steps<5*m:
            self.steps = 5*m
        for i in range(self.steps):
            '''计算最新的学习率和学习半径'''
            rate,rod = self.ratecalc(i)
            self.RateList.append(rate)
            self.RoundList.append(rod)
            '''随机选取样本，并找到优胜节点'''
            k = np.random.randint(0,m)
            tempData = normData[k,:]
            dataDist = [self.edist(tempData,a) for a in self.w.T]
            minIndex = dataDist.index(min(dataDist))
            #minIndex = np.array([self.edist(tempData,i) for i in self.w.T]).argmin()
            '''定位输出的节点位置，并计算邻域'''
            x = np.floor(minIndex/self.N)       #下取整
            y = np.mod(minIndex,self.N)         #取模
            leafDist = [self.edist(np.array([x,y]),b) for b in grid]
            #leafDist = [self.edist(grid[minIndex],b) for b in grid]
            rodIndex = list((np.array(leafDist)<rod).nonzero()[0])      #得到再学习半径范围内的输出节点下标
            for d in range(self.w.shape[1]):
                if d in rodIndex:
                    self.w[:,d] = self.w[:,d]+rate*(tempData-self.w[:,d])
        '''开始分类'''
        self.Y = np.ones(m)
        for i in range(m):
            Ydists = [self.edist(normData[i,:],j) for j in self.w.T]
            label = np.array(Ydists).argmin()
            self.Y[i] = label


'''执行程序'''
som = Kohonen()
som.loadData("D:\\mywork\\test\\ML\\4k2_far_data.txt")
som.train()
print(som.w)
print(som.Y)

'''可视化'''
newdata = np.column_stack((som.dataSet,som.Y))
plt.figure()
for i in set(newdata[:,2]):
    x = newdata[(newdata[:,2]==i).nonzero()[0],0]
    y = newdata[(newdata[:,2]==i).nonzero()[0],1]
    if i==0:
        plt.scatter(x,y,c='b',marker='o')
    elif i==1:
        plt.scatter(x,y,c='r',marker='^')
    elif i==2:
        plt.scatter(x,y,c='g',marker='h')
    elif i==3:
        plt.scatter(x,y,c='r',marker='h')
    elif i==4:
        plt.scatter(x,y,c='b',marker='D')
    else:
        plt.scatter(x,y,c='y',marker='d')
plt.show()
'''计算每一类的个数'''
for i in set(newdata[:,2]):
    print(len((newdata[:,2]==i).nonzero()[0]))

###################玻尔兹曼机##################
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

'''定义玻尔兹曼网络类'''
class BoltzmannNet(object):
    '''1、定义属性'''
    def __init__(self):
        self.dataSet = 0            #数据集
        self.Max_iter = 2000        #最大迭代次数
        self.T0 = 1000              #初始温度
        self.Lambda = 0.97          #降温速率
        self.bestIter = 0           #迭代最优时的次数
        self.dist = []              #每次迭代的距离
        self.pathindex = []         #路径的下标
        self.bestDist = 0           #最优距离
        self.bestPath = []          #最优路径
    '''2、读入数据'''
    def loadData(self,path):
        with open(path,"r") as f:
            content = f.readlines()
        self.dataSet = np.array([[float(row.strip().split()[0]),float(row.strip().split()[1])] for row in content])
        self.signs  = [row.strip().split()[2] for row in content]
    '''3、定义欧氏距离函数'''
    def eDist(self,v1,v2):
        eps = 1.0e-6
        return(np.linalg.norm(v1-v2)+eps)
    '''4、玻尔兹曼机函数'''
    def boltzmann(self,deltaX,T):
        return(np.exp(-(deltaX)/T))
    '''5、计算路径距离'''
    def distance(self,dist,path):
        N = len(path)       #路径点个数
        nowDist = 0
        for i in range(N-1):
            nowDist += dist[path[i],path[i+1]]      #路径点依次相加
        nowDist += dist[path[0],path[N-1]]          #首位点相加
        return nowDist
    '''6、改变路径函数'''
    def changepath(self,path):
        N = len(path)       #路径点个数
        '''随机产生两个位置，并交换两个位置的下标'''
        if np.random.rand() < 0.25:
            pots = np.floor(np.random.rand(1,2)*N)[0]
            newpath = copy.deepcopy(path)
            newpath[int(pots[0])] = path[int(pots[1])]
            newpath[int(pots[1])] = path[int(pots[0])]
        #'''整段位移相互转换'''
        else:
            pots = np.floor(np.random.rand(1,3)*N)[0]
            pots.sort()
            a = int(pots[0])
            b = int(pots[1])
            c = int(pots[2])
            if a!=b and b!=c:
                newpath = copy.deepcopy(path)
                newpath[a:c-1] = path[b-1:c-1] + path[a:b-1]
            else:
                newpath = self.changepath(path)
        return newpath
    '''7、初始化距离'''
    def init_bmNet(self,data):
        M = data.shape[0]
        path0 = list(range(M))
        np.random.shuffle(path0)            #打乱下标
        dist0 = self.distance(data,path0)
        self.pathindex.append(path0)
        self.dist.append(dist0)
        return(self.T0,path0,dist0)        #返回初始设定温度，随机初始路径，随机初始路径距离和
    '''8、训练主函数'''
    def train(self):
        m,n = self.dataSet.shape
        '''两两相乘，形成距离矩阵'''
        distSet = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                distSet[i,j] = self.eDist(self.dataSet[i,:],self.dataSet[j,:])
        '''首次计算距离/初始化'''
        T, path0, dist0 = self.init_bmNet(distSet)
        steps = 0
        while steps<=self.Max_iter:
            substeps = 0
            while substeps<=m:
                path1 = self.changepath(path0)
                dist1 = self.distance(distSet,path1)
                '''正常情况下：新路径距离和小于旧路径，则替代'''
                if dist1<dist0:
                    path0 = path1
                    dist0 = dist1
                    self.pathindex.append(path0)
                    self.dist.append(dist0)
                    self.bestIter+=1
                #'''对于新路径距离大于旧路径的，通过退火确定是否替换'''
                else:
                    deltaX = dist1-dist0
                    if np.random.rand()<self.boltzmann(deltaX,T):
                        path0 = path1
                        dist0 = dist1
                        self.pathindex.append(path0)
                        self.dist.append(dist0)
                        self.bestIter+=1
                substeps += 1
            steps += 1
            T = T*self.Lambda
        '''取出最优路径'''
        self.bestDist = min(self.dist)
        self.bestPath = self.pathindex[np.argmin(self.bestDist)]

'''正式计算最短路径问题'''
bmNet = BoltzmannNet()
path = "D:\\mywork\\test\\ML\\dataSet25_Boltzmann.txt"
bmNet.loadData(path)
data = bmNet.dataSet
bmNet.train()
'''最优解'''
print(bmNet.bestDist)
print(bmNet.bestPath)
'''可视化'''
paths = bmNet.pathindex
dists = bmNet.dist
iters = bmNet.bestIter
'''路径距离变化可视化'''
plt.figure()
plt.plot(range(iters+1),dists)
plt.show()
'''最优路径可视化'''
signs = bmNet.signs
x = [data[i,0] for i in bestPath]
y = [data[i,1] for i in bestPath]
s = [signs[i] for i in bestPath]
plt.figure()
plt.scatter(x,y,c='r',linewidths=5)
i=0
for xl,yl in zip(x,y):
    plt.annotate("%s" %s[i], xy=(xl+50,yl+50))
    i+=1
plt.plot(x,y,'b--')
plt.show()




