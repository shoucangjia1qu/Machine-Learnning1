# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:01:48 2019

@author: ZWD
"""

import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r"D:\mywork\test")

#用SMO算法求得最小值
class SMO(object):
    #1、类的属性
    def __init__(self):
        self.Xdata = 0              #测试集数据
        self.Ylabel = 0             #数据实际标签
        self.alpha = 0              #参数
        self.C = 0                  #惩罚因子
        self.Kvalue = dict()        #核函数类型和参数
        self.K = 0                  #核技巧之后的结果
        self.maxiters = 0           #最大循环次数
        self.tol = 0                #容错率
        self.b = 0                  #截距
        self.m = 0                  #样本数量
        self.svindex = 0            #支持向量的下标
        self.SVMvects = 0           #支持向量
        self.SVMlabels = 0          #支持向量的标签
        self.EiMark = 0             #误差是否更新的标识
    
    #2、核函数    
    def kernel(self,X,Z):
        if list(self.Kvalue.keys())[0] == 'Linear':
            Ki = np.dot(X, Z.T)
        elif list(self.Kvalue.keys())[0] == 'Gaussian':
            L2 = np.power(np.linalg.norm((X-Z), axis=1),2)
            #np.power((X-Z),2).sum(axis=1)
            Ki = np.exp(L2/(-1*self.Kvalue['Gaussian']**2))
        else:
            raise NameError('无法识别的核函数')
        return Ki
    
    
    #3、计算分离超平面的距离
    def distHyperplane(self,i):
        return np.dot(self.K[i,:], np.multiply(self.alpha, self.Ylabel)) + self.b
    
    #4、计算误差
    def calError(self,i):
        Ei = self.distHyperplane(i) - self.Ylabel[i]
        return Ei
    
    #5、SMO算法中第二个参数的修剪
    def cut(self,alpha2New,L,H):
        if alpha2New < L:
            alpha2New = L
        elif alpha2New > H:
            alpha2New = H
        else:
            pass
        return alpha2New
    
    #6、SMO算法中第二个参数上下界的选择
    def LandH(self,y1,y2,alpha1Old,alpha2Old):
        if y1 != y2:
            L = max(0, alpha2Old-alpha1Old)
            H = min(self.C, self.C+alpha2Old-alpha1Old)
        else:
            L = max(0, alpha2Old+alpha1Old-self.C)
            H = min(self.C, alpha2Old+alpha1Old)
#        print('L:',L,'H:',H)
        return L, H
    
    #7、初始化参数
    def initParameter(self):
        self.alpha = np.zeros((self.m,1))
        self.EiMark = np.zeros((self.m,1))
        self.K = np.zeros((self.m,self.m))
        for i in range(self.m):
            self.K[i,:] = self.kernel(self.Xdata, self.Xdata[i])
        return
    
    #8-1、寻找第二个参数，寻找Max(E1-E2)，方法1：随机选取第一个E2
    def choosei2(self,i1):
        self.EiMark[i1] = 1
        E1 = self.calError(i1)
        maxDeltaE = 0
        besti2 = 0
        if np.sum(self.EiMark, axis=0) == 1:
            while True:
                besti2 = np.random.randint(0,self.m)
                if besti2 != i1:
                    self.EiMark[besti2] = 1
#                    print('随机选取{}'.format(besti2))
                    break
        else:
            for i in range(self.m):
                if i == i1:
                    continue
                E2 = self.calError(i)
                DeltaE = abs(E1-E2)
                if DeltaE > maxDeltaE:
                    maxDeltaE = DeltaE
                    besti2 = i
#        print('最大差法{}'.format(besti2))
        return besti2
        
    
    #8-2、寻找第二个参数，寻找MAX(E1-E2)，方法2：基本固定下来
#    def choosei2(self,i1):
#        E1 = self.calError(i1)
#        maxDeltaE = 0
#        besti2 = 0
#        for i in range(self.m):
#            if i == i1:
#                continue
#            E2 = self.calError(i)
#            DeltaE = abs(E1-E2)
#            if DeltaE > maxDeltaE:
#                maxDeltaE = DeltaE
#                besti2 = i
#            if i1<5:
#                print('E1:',E1,'E2:',E2,'besti2:',besti2)
#        return besti2
    
    #9、内循环：更新迭代alpha1和alpha2参数，和截距b
    def alphaUpdate(self,i1, i2):
        #9-1 两个样本的标签
        y1 = self.Ylabel[i1]
        y2 = self.Ylabel[i2]
        #9-2 参数
        alpha1Old = self.alpha[i1,0].copy()
        alpha2Old = self.alpha[i2,0].copy()
        #9-3 误差
        E1 = self.calError(i1)
        E2 = self.calError(i2)
        #9-4 核转换后的结果
        K11 = self.K[i1,i1]
        K22 = self.K[i2,i2]
        K12 = self.K[i1,i2]
        #结束条件1：
        
        if (K11+K22-2*K12)<=0: return 0
        #9-5 第二个参数的上下界
        L, H = self.LandH(y1,y2,alpha1Old,alpha2Old)
        #结束条件2：
        
        if L == H: return 0
        #9-6 更新第二个参数
        alpha2New = alpha2Old + y2*(E1-E2)/(K11+K22-2*K12)
        #9-7 剪切第二个参数
        alpha2New = self.cut(alpha2New,L,H)
        #结束条件3：
#        print(3)
        if abs(alpha2New - alpha2Old) < 1.0e-5: return 0
        #9-8 更新第一个参数
        alpha1New = alpha1Old + (alpha2Old-alpha2New)*y1*y2
        self.alpha[i1] = alpha1New; self.alpha[i2] = alpha2New
        #9-9 更新截距b
        b1 = self.b - E1 - (alpha1New-alpha1Old)*y1*K11 - (alpha2New-alpha2Old)*y2*K12
        b2 = self.b - E2 - (alpha2New-alpha2Old)*y1*K22 - (alpha1New-alpha1Old)*y2*K12
        if (0<alpha1New and alpha1New<self.C):
            self.b = b1
        elif (0<alpha2New and alpha2New<self.C):
            self.b = b2
        else:
            self.b=(b1+b2)/2
        #结束条件4：
#        print('旧alpha1',alpha1Old,'旧alpha2',alpha2Old,'新alpha1',alpha1New,'新alpha2',alpha2New,'b',self.b)
        return 1
    
    #10、主函数：选择第一个变量进行循环
    def train(self, X, Y, C, kernel, kernalParameter, maxIters=10, tol=0.001):
        #10-1对属性进行赋值
        self.Kvalue[kernel] = kernalParameter
        self.maxiters = maxIters
        self.tol = tol
        self.C = C
        self.Xdata = X
        self.Ylabel = Y
        m, n =np.shape(X)
        self.m = m
        #10-2初始化参数，拉格朗日乘子、核技巧结果等
        self.initParameter()     
        #10-3选择第一个参数
        flag = True             #第一个参数是否遍历全部数据集
        mark = 0                #拉格朗日乘子修改次数
        step = 0                #迭代次数
        while (step<self.maxiters) and (mark>0 or flag):
            mark = 0
            if flag:
                SVMidx = list(range(m))                                         #2、若(0,C)之间的点都符合KKT条件，则遍历全部
                for idx in SVMidx:
                    alpha1 = self.alpha[idx,0]
                    Gi = self.distHyperplane(idx)
                    yi = self.Ylabel[idx,0]
                    if (alpha1==0 and (yi*Gi+self.tol)<1) or (alpha1==self.C and (yi*Gi+self.tol)>1) or (alpha1>0 and alpha1<self.C and (yi*Gi)!=1):
                        i1 = np.copy(idx)                       #找到第1个参数
                        i2 = self.choosei2(i1)                       #找到第2个参数                  
#                        print('i1:',i1,'i2:',i2)
#                        print(">>")
                        mark += self.alphaUpdate(i1, i2)        #进行内循环
                step += 1
                print('全量数据集，第{}轮：，内循环{}次'.format(step,mark))
                print('==================================================')
            else:
                SVMidx = np.nonzero((self.alpha>0)*(self.alpha<self.C))[0]      #1、优先寻找(0,C)之间的点，看是否满足KKT条件
                for idx in SVMidx:
                    alpha1 = self.alpha[idx,0]
                    Gi = self.distHyperplane(idx)
                    yi = self.Ylabel[idx,0]
                    if (alpha1==0 and (yi*Gi+self.tol)<1) or (alpha1==self.C and (yi*Gi+self.tol)>1) or (alpha1>0 and alpha1<self.C and (yi*Gi)!=1):
                        i1 = np.copy(idx)                       #找到第1个参数
                        i2 = self.choosei2(i1)                       #找到第2个参数
#                        print('i1:',i1,'i2:',i2)
#                        print(">>")
                        mark += self.alphaUpdate(i1, i2)        #进行内循环
                step += 1
                print('(0,C)数据集，第{}轮：，内循环{}次'.format(step,mark))
                print('==================================================')
            if flag: flag=False                             #切换到(0,C)之间的点
            elif (mark == 0): flag = True                   #切换到全部数据集
        self.svindex = np.nonzero(self.alpha>0)[0]
        self.SVMvects = self.Xdata[self.svindex]
        self.SVMlabels = self.Ylabel[self.svindex]
        
    #11、预测
    def predict(self,testSet):
        return np.sign(self.distHyperplane(testSet))
        
        
        
        
#读取数据集，并画图
with open(r'D:\mywork\test\ML\郑捷《机器学习算法原理与编程实践》第2-9章节的源代码及数据集\支持向量机\svm.txt', 'r') as f:
    d = f.readlines()
data = np.array([[float(comment) for comment in row.split()] for row in d])
X = data[:,:2]
Y = data[:,2].reshape((len(data),1))        
plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b',marker='D')      
plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r')        
plt.show()

#训练
smo = SMO()        
smo.train(X, Y, 100, 'Gaussian', 3, maxIters=20)
SVMidx = smo.svindex        
SVMvects = smo.SVMvects
SVMlabels = smo.SVMlabels
print(len(SVMidx))


plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1],c='b')      
plt.scatter(X[Y[:,0]==-1,0],X[Y[:,0]==-1,1],c='r')
plt.scatter(SVMvects[SVMlabels[:,0]==1,0],SVMvects[SVMlabels[:,0]==1,1],c='g',marker='D',linewidths=5)
plt.scatter(SVMvects[SVMlabels[:,0]==-1,0],SVMvects[SVMlabels[:,0]==-1,1],c='y',marker='D',linewidths=5)        
plt.show()
        
n=0
for i in range(200):
    pre = np.sign(smo.distHyperplane(i))
    if pre!=smo.Ylabel[i]:
        n += 1
        print("第{}个，应该是{}".format(i,smo.Ylabel[i]))        
    


#之前的SMO算法
class PlattSVM(object):
    def __init__(self):
        self.trainSet = 0       #数据集
        self.Labels = 0         #标签
        self.K = 0              #经核函数转变后的点积
        self.kValue = dict()    #核函数的参数
        self.C = 0              #惩罚因子C
        self.alpha = 0          #拉格朗日乘子
        self.tol = 0            #容错率
        self.maxiters = 100     #最大循环次数
        self.b = 0              #截距初始值
        
        
    
    def loadData(self,filename):
        with open(filename,"r") as f:
            content = f.readlines()
            OriData = np.array([[float(comment) for comment in row.split()] for row in content])
        self.trainSet = OriData[:,:2]
        self.Labels = OriData[:,2].reshape((len(OriData),1))
    
    def initparam(self):
        m,n = np.shape(self.trainSet)
        self.alpha = np.zeros((m,1))
        self.eCache = np.zeros((m,2))          #一列为标注误差是否更新，一列是记载的误差
        self.K = np.zeros((m,m))
        for k in range(m):
            self.K[k,:] = self.kernels(self.trainSet,self.trainSet[k,:])
    
    def kernels(self,data,A):
        if list(self.kValue.keys())[0] == "linear":
            Ki = np.dot(data,A.T)
        elif list(self.kValue.keys())[0] == "Gaussian":
            x = np.power((data - A),2)
            Mo = np.sum(x,axis=1)
            Ki = np.exp(Mo/(-1*self.kValue['Gaussian']**2))
        else:
            raise NameError('无法识别的核函数')
        
        return Ki

    def calEk(self,i):
        Ek = float(np.dot(np.multiply(self.alpha,self.Labels).T,self.K[:,i])) + self.b - self.Labels[i,0]      # Yp=W*X+b;E=Yp-Y
        return Ek
    
    def chooseJ(self,i,Ei):
        maxj = -1               #取0的话容易变成0.13
        maxEj = 0
        maxDeltaE = 0
        self.eCache[i] = (1,Ei)
        SvmError = np.nonzero(self.eCache[:,0])[0]         #判断当前误差的个数，如果只有一个说明是第一次，就可以随机选取J
        if len(SvmError)>1:
            for j in SvmError:
                if j==i:
                    continue
                Ej = self.calEk(j)                   #重新遍历第二个变量的误差
                deltaE = abs(Ei-Ej)                  #第一个和第二个变量的误差差
                if deltaE>maxDeltaE:
                    maxj = j                        #最终选取的第二个变量
                    maxEj = Ej                      #最终选取的第二个变量的误差
                    maxDeltaE = deltaE
            return maxj,maxEj
        else:
            while True:
                j = np.random.randint(0,np.shape(self.trainSet)[0])
                if j != i :
                    break
            Ej = self.calEk(j)
            return j,Ej
    
    def cutalpha(self,alpha,L,H):
        if alpha>H:
            alpha = H
        if alpha<L:
            alpha = L
        return alpha
    
    def train(self):
        self.initparam()        #初始化
        m,n = np.shape(self.trainSet)
        step = 0                #循环次数
        flag = True             #主循环标识
        AlphaChange = 0         #内循环标识
        while step<self.maxiters and (flag or AlphaChange>0):
            AlphaChange = 0         #内循环标识
            if flag:
                for i in range(m):                          #2、其次遍历全量数据集
                    Ei = self.calEk(i)
                    alpha1 = self.alpha[i,0]
                    y1 = self.Labels[i,0]
                    if (alpha1==0 and y1*Ei<-self.tol) or (alpha1==self.C and y1*Ei>self.tol) or (alpha1>0 and alpha1<self.C and y1*Ei!=0):
#                        print('i1:',i)
                        AlphaChange += self.inner(i,Ei,y1)        #内循环返还标识
                print("全量",step,AlphaChange)
                step+=1
            else:
                SvmAlpha = np.nonzero((self.alpha>0)*(self.alpha<self.C))[0]        #1、优先查找（0，C）之间的拉格朗日乘子
                for i in SvmAlpha:
                    Ei = self.calEk(i)
                    alpha1 = self.alpha[i,0]
                    y1 = self.Labels[i,0]
                    if (alpha1==0 and y1*Ei<-self.tol) or (alpha1==self.C and y1*Ei>self.tol) or (alpha1>0 and alpha1<self.C and y1*Ei!=0):
#                        '''(alpha1<self.C and y1*Ei<-self.tol) or (alpha1>0 and y1*Ei>self.tol)'''
                        AlphaChange += self.inner(i,Ei,y1)        #内循环返还标识
                print("KKT",step,AlphaChange)
                step+=1
            if flag : flag = False                                                                                          #转换标志位：切换到另一种
            elif (AlphaChange == 0) :flag = True                            #改变主循环数据集为（0，C）
        self.svIdx = np.nonzero(self.alpha>0)[0]            #支持向量的下标
        self.sptVects = self.trainSet[self.svIdx]           #支持向量
        self.SVlabels = self.Labels[self.svIdx]             #支持向量的标签
        print(step)

    def inner(self,i,Ei,y1):
        j,Ej = self.chooseJ(i,Ei)           #生成第二个变量
#        print('i2',j)
        y2 = self.Labels[j,0]
        oldAlpha1 = self.alpha[i,0].copy()     #生成旧的第一个alpha
        oldAlpha2 = self.alpha[j,0].copy()     #生成旧的第二个alpha
#        print("================================")
#        print("i:{},lambdaI:{},Ei:{}".format(i,oldAlpha1,Ei))
#        print("j:{},lambdaJ:{},Ej:{}".format(j,oldAlpha2,Ej))
        if y1!=y2:
            L=max(0,oldAlpha2-oldAlpha1)
            H=min(self.C,self.C+oldAlpha2-oldAlpha1)
        else:
            L=max(0,oldAlpha2+oldAlpha1-self.C)
            H=min(self.C,oldAlpha2+oldAlpha1)
        if L==H:
            return 0
        eta = self.K[i,i] + self.K[j,j] - 2*self.K[i,j]
        if eta<=0:
            return 0
        self.alpha[j,0] = oldAlpha2 +y2*(Ei-Ej)/eta
        self.alpha[j,0] = self.cutalpha(self.alpha[j,0],L,H)
        if abs(oldAlpha2-self.alpha[j,0])<0.00001:
            return 0
        self.alpha[i,0] = oldAlpha1 + y1*y2*(oldAlpha2-self.alpha[j,0])
        self.eCache[j] = (1,self.calEk(j))
        self.eCache[i] = (1,self.calEk(i))
        b1 = self.b-Ei-y1*self.K[i,i]*(self.alpha[i,0]-oldAlpha1)-y2*self.K[j,i]*(self.alpha[j,0]-oldAlpha2)
        b2 = self.b-Ej-y1*self.K[i,j]*(self.alpha[i,0]-oldAlpha1)-y2*self.K[j,j]*(self.alpha[j,0]-oldAlpha2)
        if (0<self.alpha[i,0] and self.alpha[i,0]<self.C):
            self.b = b1
        elif (0<self.alpha[j,0] and self.alpha[j,0]<self.C):
            self.b = b2
        else:
            self.b=(b1+b2)/2
        return 1

    def scatterplot(self, plt):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(np.shape(self.trainSet)[0]) :
            if self.alpha[i] != 0 :
                ax.scatter(self.trainSet[i, 0], self.trainSet[i, 1], c = 'green', marker = 's')
            elif self.Labels[i] == 1 :
                ax.scatter(self.trainSet[i, 0], self.trainSet[i, 1], c = 'blue', marker = 'o')
            elif self.Labels[i] == -1 :
                ax.scatter(self.trainSet[i, 0], self.trainSet[i, 1], c = 'red', marker = 'o')

    
    def predict(self,testSet):
        m,n = np.shape(testSet)
        preLabels = np.zeros([m,1])
        for i in range(m):
            sigmaK = self.kernels(self.sptVects,testSet[i,:])
            preY = np.multiply(self.alpha[self.svIdx],self.SVlabels).T*sigmaK + self.b
            preLabels[i,0] = np.sign(preY)
        return preLabels

    def classify(self, testSet, testLabel) :
        errorCount = 0
        testMat = mat(testSet)
        m, n = shape(testMat)
        for i in range(m) :
            kernelEval = self.kernels(self.sptVects, testMat[i, :])
            predict = kernelEval.T * multiply(self.SVlabels, self.alpha[self.svIdx]) + self.b
            if sign(predict) != sign(testLabel[i]) : errorCount += 1
        return float(errorCount) / float(m)



svm = PlattSVM()
svm.C = 100                                                                                   #惩罚因子
svm.tol = 0.001                                                                            #容错律
svm.maxiters = 100
svm.kValue['Gaussian'] = 3.0                                                  #核函数
svm.loadData('svm.txt')
svm.train()



#课后习题6.2
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
X = np.array(dataSet)[:,6:8]
X=X.astype(float)
Y = np.array(dataSet)[:,8]
Y[Y=="好瓜"]=1
Y[Y=="坏瓜"]=-1
Y=Y.astype(float).reshape(-1,1)
#训练
smo = SMO()
smo.train(X, Y, 100, 'Gaussian', 1.3, maxIters=20)
SVMidx = smo.svindex        
SVMvects = smo.SVMvects
SVMlabels = smo.SVMlabels
#预测
testSet = X
m, n = testSet.shape
preLabels = np.zeros([m,1])
for i in range(m):
    sigmaK = smo.kernel(SVMvects,testSet[i,:])
    preY = np.dot(np.multiply(smo.alpha[SVMidx],SVMlabels).T, sigmaK.reshape(-1,1)) + smo.b
    preLabels[i,0] = np.sign(preY)


#UCI数据集测试
from sklearn import preprocessing
from sklearn import model_selection

with open("UCI_data\\iris.data") as f:
    iris_data = f.readlines()
iris_data = [row.split(',') for row in iris_data][:-1]
m, n = np.shape(iris_data)
iris = np.zeros((m, n-1))
irislabel = []
for i in range(m):
    iris[i,:] = iris_data[i][:-1]
    irislabel.append(iris_data[i][-1])
yencoder = preprocessing.LabelEncoder()
irislabel = yencoder.fit_transform(irislabel)    
X = iris[:100]
Y = irislabel[:100]
Y[Y==0] = -1
trainx, testx, trainy, testy = model_selection.train_test_split(X, Y, train_size=0.8, random_state=1234)
trainy = trainy.reshape(-1,1)
testy = testy.reshape(-1,1)
Y = Y.reshape(-1,1)
#训练
smo = SMO()
smo.train(trainx, trainy, 100, 'Gaussian', 3, maxIters=20)
SVMidx = smo.svindex        
SVMvects = smo.SVMvects
SVMlabels = smo.SVMlabels
#预测
m, n = X.shape
preLabels = np.zeros([m,1])
for i in range(m):
    sigmaK = smo.kernel(SVMvects,X[i,:])
    preY = np.dot(np.multiply(smo.alpha[SVMidx],SVMlabels).T, sigmaK.reshape(-1,1)) + smo.b
    preLabels[i,0] = np.sign(preY)
sum(Y != preLabels)

