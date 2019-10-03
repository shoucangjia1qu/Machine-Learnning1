# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 21:43:43 2019

@author: ecupl
"""
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r"D:\mywork\test")

#高斯混合聚类模型类
class GaussianMix(object):
    #1、类的属性
    def __init__(self):
        self.trainSet = 0               #数据集
        self.ClusterLabel = 0           #聚类标签
        self.k = 0                      #聚类个数
        self.LL_ValueList = []          #最大似然函数的值列表
        self.AlphaArr = 0               #高斯混合模型混合系数
        self.MiuArr = 0                 #高斯分布函数的均值参数
        self.SigmaArr = 0               #高斯分布函数的协方差参数
        self.m = 0                      #样本数
        self.d = 0                      #样本维度
        
    #2、初始化函数参数
    def initParas(self, x, k):
        """
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            k:需要聚类的个数
        return:
            初始化AlphaArr, MiuArr, SigmaArr
        """
        self.trainSet = x
        self.k = k
        self.m, self.d = np.shape(x)
        AlphaArr0 = np.ones((1,k))/k
        MiuArr0 = x[np.random.randint(0, self.m, k)]
        #在这里固定好了
        #MiuArr0 = x[[5,6,12]]
        SigmaArr0 = np.array([(np.eye(self.d)*0.1).tolist()]*k)
        return AlphaArr0, MiuArr0, SigmaArr0

    
    #3、计算高斯分布函数
    def Gaussian_multi(self, x, miu, sigma):
        """
        多元高斯分布的密度函数
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            miu:该高斯分布的均值,1*d维
            sigma:该高斯分布的标准差,在此为d*d的协方差矩阵
        return:
            distributionArr:返回样本的概率分布1D数组
        """
        distributionArr = np.exp(-0.5*np.sum(np.multiply(np.dot(x-miu, np.linalg.pinv(sigma)), x-miu), axis=1))/\
        (np.power(2*np.pi, 0.5*self.d)*np.linalg.det(sigma)**0.5)
        return distributionArr

    #4、计算观测值y，高斯分布函数参数条件下，观测来自于第k个高斯分布的概率
    def Gama_Prob(self, x, AlphaArr, MiuArr, SigmaArr):
        """
        计算当观测值已知，是哪个高斯模型产品该观测值的概率
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            AlphaArr:每个高斯模型出现的先验概率,1*k维,k为聚类个数
            MiuArr:每个高斯模型的均值参数,k*d维
            SigmaArr:每个高斯模型的协方差矩阵参数,k*d*d维
        return:
            GamaProbArr:每个样本出现对应每个高斯模型分布概率的矩阵,m*k维
        """
        GaussProbArr = np.zeros((self.m, self.k))
        for i in range(self.k):
            miu = MiuArr[i]
            sigma = SigmaArr[i]
            GaussProbArr[:,i] = self.Gaussian_multi(x, miu, sigma)
        GamaProbArr = np.copy(np.multiply(GaussProbArr, AlphaArr))
        SumGamaProb = np.sum(GamaProbArr, axis=1).reshape(-1,1)
        return (GamaProbArr/SumGamaProb).round(4), GamaProbArr.round(4)
    
    #5、更新高斯分布函数参数
    def updateParas(self, x, GamaProbArr):
        """
        更新高斯分布函数的参数，包括均值和协方差矩阵
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            GamaProbArr:高斯分布函数的后验概率,m*k维
        return:
            newMiuArr:更新后的高斯分布函数的均值,k*d维
            newSigmaArr:更新后的高斯分布的协方差矩阵,k*d*d维
            newAlphaArr:更新后的高斯模型的混合系数,1*k维
        """
        SumGamaProb = np.sum(GamaProbArr, axis=0)
        newMiuArr = np.zeros((self.k,self.d))
        newSigmaArr = np.zeros((self.k,self.d,self.d))
        for i in range(self.k):
            Gama = GamaProbArr[:,i].reshape(-1,1)
            #更新均值
            newMiu = np.sum(np.multiply(Gama, x), axis=0)/SumGamaProb[i]
            newMiuArr[i] = newMiu
            #更新协方差矩阵
            newSigma = np.dot(np.multiply(x-newMiu, Gama).T, x-newMiu)/SumGamaProb[i]
            newSigmaArr[i] = newSigma
        newAlphaArr = SumGamaProb.reshape(1,-1)/self.m
        return newMiuArr, newSigmaArr, newAlphaArr

    #6、求似然函数值
    def calLLvalue(self, GamaProbArr, GaussProbArr):
        lnGaussProbArr = np.log(GaussProbArr+1.0e-6)
        LLvalue = np.sum(np.multiply(GamaProbArr, lnGaussProbArr))
        return LLvalue
    
    #7、训练：判断是否符合停止条件
    def train(self, x, k, iters):
        """
        循环迭代
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            k:聚类个数
            iters:迭代次数
        return:
            ClusterLabel:最终的聚类结果
        """
        #初始化参数
        AlphaArr0, MiuArr0, SigmaArr0 = self.initParas(x, k)
        LLvalue0 = 0                #初始似然函数值
        LLvalueList = []            #最大似然值列表
        for i in range(iters):
            #计算高斯分布模型的后验概率，也就是已知观测下来自于第k个高斯分布函数的概率
            GamaProbArr, GaussProbArr = self.Gama_Prob(x, AlphaArr0, MiuArr0, SigmaArr0)
            #计算聚类结果
            ClusterLabel = np.argmax(GamaProbArr, axis=1)
            #画分布图(适用于二维)
            #self.drawPics(x, MiuArr0, SigmaArr0, ClusterLabel)
            #计算似然函数，并判断是否继续更新
            #LLvalue1 = self.calLLvalue(GamaProbArr, GaussProbArr)
            LLvalue1 = sum(np.log(GaussProbArr.sum(axis=1)+1.0e-6))
            print('似然值：',LLvalue1)
            if len(LLvalueList) == 0:
                LLvalue0 = LLvalue1
            else:
                LLdelta = LLvalue1 - LLvalue0
                print('似然值提升：',LLdelta)
                if abs(LLdelta)<1.0e-6:
                    break
                else:
                    LLvalue0 = LLvalue1
            LLvalueList.append(LLvalue1)
            #继续迭代，更新函数参数
            MiuArr1, SigmaArr1, AlphaArr1 = self.updateParas(x, GamaProbArr)
            MiuArr0 = np.copy(MiuArr1)
            SigmaArr0 = np.copy(SigmaArr1)
            AlphaArr0 = np.copy(AlphaArr1)
        self.ClusterLabel = ClusterLabel
        self.LL_ValueList = LLvalueList
        plt.plot(range(len(LLvalueList)), LLvalueList)
        plt.show()
        return

    #8-1、画图
    def drawPics(self, x, MiuArr, SigmaArr, Clusters):
        """
        画图，不同聚类类别的点分布，等高图
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            MiuArr, SigmaArr:高斯分布函数的参数
            Clusters:聚类结果
        out:
            散点图+等高分布图
        """
        plt.figure(figsize=(10,6))
        xgrid, ygrid, zgrid = self.calXYZ(x, MiuArr, SigmaArr)
        #c=plt.contour(xgrid,ygrid,zgrid,6,colors='black')
        plt.contour(xgrid,ygrid,zgrid,6,colors='black')
        plt.contourf(xgrid,ygrid,zgrid,6,cmap=plt.cm.Blues,alpha=0.5)
        #plt.clabel(c,inline=True,fontsize=10)
        for i in range(self.k):
            xi = x[Clusters==i,0]
            yi = x[Clusters==i,1]
            plt.scatter(xi, yi)
            plt.scatter(MiuArr[i,0], MiuArr[i,1], c='r', linewidths=5, marker='D')
        plt.show()
        return

    #8-2、计算网格状的高斯分布，用于画等高线
    def calXYZ(self, x, MiuArr, SigmaArr):
        """
        画等高图需要计算X,Y,Z
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            MiuArr, SigmaArr:高斯分布函数的参数
        return:
            xgrid:x的网格坐标
            ygrid:y的网格坐标
            zgrid:(x,y)网格坐标上高斯分布函数的概率
        """
        x1 = np.copy(x[:,0])
        x1.sort()
        y1 = np.copy(x[:,1])
        y1.sort()
        x2,y2 = np.meshgrid(x1,y1)  # 获得网格坐标矩阵
        Gp = np.zeros((self.m,self.m))
        for i in range(self.m):
            for j in range(self.m):
                xi = x2[i,j]
                yi = y2[i,j]
                data = np.array([xi,yi])
                miuList=[]
                for miu, sigma in zip(MiuArr, SigmaArr):   
                    p = np.exp(-0.5*np.dot(np.dot((data-miu).reshape(1,-1), np.linalg.inv(sigma)), (data-miu).reshape(-1,1)))/\
                    np.power(2*np.pi, 0.5*self.d)*np.linalg.det(sigma)**0.5
                    miuList.append(p)
                Gp[i,j] = max(miuList)
        return x2, y2, Gp
    
#开始训练
if __name__ == "__main__":
    ##############西瓜集数据4.0
    data = np.array([[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],[0.403,0.237],[0.481,0.149],
                     [0.437,0.211],[0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],
                     [0.360,0.370],[0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],[0.748,0.232],
                     [0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],[0.751,0.489],[0.532,0.472],[0.473,0.376],
                     [0.725,0.445],[0.446,0.459]])
    GMM = GaussianMix()
    k = 3
    GMM.train(data, k, 50)
    Clusters = GMM.ClusterLabel
    labels = GMM.ClusterLabel
    ##############鸢尾花数据，利用聚类结果和实际标签进行比较
    with open(r"D:\mywork\test\UCI_data\iris.data") as f:
        data = f.readlines()
    trainSet = np.array([row.split(',') for row in data[:-1]])
    trainSet = trainSet[:,:-1].astype('float')
    trainSet = (trainSet - trainSet.mean(axis=0))/trainSet.std(axis=0)
    k = 3
    GMM = GaussianMix()
    GMM.train(trainSet, k, 50)
    Clusters = GMM.ClusterLabel
    labels = GMM.ClusterLabel




#评价指标        
##轮廓系数
def LK(train,Labels):
    LK = []
    m = 0
    for data in train:
        n=0
        a = 0
        b = dict()
        avalue = 0
        bvalue = 0
        for subdata in train: 
            if m==n:
                n += 1
                continue
            if Labels[m] == Labels[n]:
                a += db.calDist(data,subdata)
            else:
                if Labels[n] not in b.keys():
                    b[Labels[n]] = 0
                b[Labels[n]] += db.calDist(data,subdata)
            n += 1
        '''a是点到本簇中其他点的平均距离'''
        avalue = (a/(len(np.nonzero(Labels==Labels[m])[0])-1))
        '''b是点到其他簇中其他点的平均距离的最小值'''
        bvalue = np.min([value/len(np.nonzero(Labels==la)[0]) for la,value in b.items()])
        LK.append((bvalue-avalue)/max(bvalue,avalue))
        m += 1
    LKratio = np.mean(LK)
    return(LKratio)
print(LK(data,labels))

##DB指数
###1、欧式距离
def calEdist(v1, v2):
    return np.linalg.norm((v1-v2))

###2、簇内平均距离
def avgCluster(x):
    dist = 0
    m, d = np.shape(x)
    for i in range(m):
        for j in range(m):
            if j > i:
                dist += calEdist(x[i], x[j])
    return dist/(m*(m-1))

###3、簇内最大距离
def maxCluster(x):
    maxdist = 0
    m, d = np.shape(x)
    for i in range(m):
        for j in range(m):
            if j > i:
                dist = calEdist(x[i], x[j])
            else:
                continue
            if dist > maxdist:
                maxdist = copy.deepcopy(dist)
    return maxdist

###4、两簇间中心点距离
def centerClusters(x1, x2):
    m1, d1 = np.shape(x1)
    m2, d2 = np.shape(x2)
    miu1 = np.sum(x1, axis=0)/m1
    miu2 = np.sum(x2, axis=0)/m2
    return calEdist(miu1, miu2)

###5、两簇间最近样本距离
def minClusters(x1, x2):
    mindist = np.inf
    m1, d1 = np.shape(x1)
    m2, d2 = np.shape(x2)
    for i in range(m1):
        for j in range(m2):
            dist = calEdist(x1[i], x2[j])
            if dist < mindist:
                mindist = copy.deepcopy(dist)
    return mindist

###6、DB指数
def calDBI(train, label):
    labelSet = np.unique(label)
    k = len(labelSet)
    DBIList = []
    for i in labelSet:
        maxDBI = 0
        for j in labelSet:
            if i==j:
                continue
            xi = train[label==i]
            xj = train[label==j]
            DBI = (avgCluster(xi) + avgCluster(xj))/centerClusters(xi, xj)
            if DBI > maxDBI:
                maxDBI = copy.deepcopy(DBI)
        DBIList.append(maxDBI)
    return sum(DBIList)/k, DBIList

print(calDBI(data, labels))

##Dunn指数
def calDI(train, label):
    labelSet = np.unique(label)
    k = len(labelSet)
    clusterMax = 0              #计算簇内样本间最大距离
    for i in labelSet:
        xl = train[label==i]
        clusterDist = maxCluster(xl)
        if clusterDist > clusterMax:
            clusterMax = copy.deepcopy(clusterDist)
    minOutCluster = []
    for i in labelSet:
        minInCluster = np.inf
        for j in labelSet:
            if i==j:
                continue
            xi = train[label==i]
            xj = train[label==j] 
            Incluster = minClusters(xi, xj)/clusterMax
            if Incluster < minInCluster:
                minInCluster = copy.deepcopy(Incluster)
        minOutCluster.append(minInCluster)
    return min(minOutCluster), minOutCluster
            
print(calDI(data, labels))






###GMM半监督学习——生成式方法
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r"D:\mywork\test")

class GaussianMix(object):
    #1、类的属性
    def __init__(self):
        self.Dl_Set = 0                 #已知标签的数据集
        self.Dl_Label = 0               #已知标签的数据集的标签
        self.Dl_Yset = 0                #已知标签的类别
        self.Du_Set = 0                 #未知标签的数据集
        self.Dl_Label = 0               #未知标签的数据集的目标标签
        self.k = 0                      #聚类个数，应该是和标签类别数一致
        self.LL_ValueList = []          #最大似然函数的值列表
        self.AlphaArr = 0               #高斯混合模型混合系数
        self.MiuArr = 0                 #高斯分布函数的均值参数
        self.SigmaArr = 0               #高斯分布函数的协方差参数
        self.l = 0                      #已知标签的数据样本数
        self.u = 0                      #未知标签的数据样本数
        self.d = 0                      #样本维度
        
    #2、初始化函数参数，原来无标记的时候是随机选取，现在可通过有标记的数据进行初始化
    def initParas(self, X, Y):
        """
        input:
            X:样本集,m*d,其中m为样本数,d为样本维度数
            Y:样本标签
        return:
            初始化k(高斯模型个数，理论上和分类数量一致？),AlphaArr, MiuArr, SigmaArr
        """
        self.l = []
        Dl_Y = np.unique(Y)
        m, d = X.shape
        k=len(Dl_Y)
        self.Dl_Yset = Dl_Y             #赋值每一类标签的值
        self.d = d                      #样本维度
        self.k = k                      #高斯模型个数
        #2-1 按照已知标签的样本计算高斯模型的参数初始值
        AlphaArr0 = np.ones((1,k))/k                        #高斯模型混合系数初始值
        MiuArr0 = np.zeros((k,d))                           #高斯模型miu参数初始值
        SigmaArr0 = np.array([(np.eye(d)).tolist()]*k)      #高斯模型sigma参数初始值
        for idx, value in enumerate(Dl_Y):
            Dl_Xi = X[np.nonzero(Y==value)[0],:]
            MiuArr0[idx] = Dl_Xi.mean(axis=0)
            Count_Xi = len(Dl_Xi)
            self.l.append(Count_Xi)
            SigmaArr0[idx] = np.dot((Dl_Xi - MiuArr0[idx]).T,(Dl_Xi - MiuArr0[idx]))/Count_Xi
        #2-2 如果是二维的数据，可以把初始的分布用等高图画出来
        #self.drawPics(X, MiuArr0, SigmaArr0, Y)
        #print("==================以上是已标记数据初始化高斯函数参数分布图==================")
        return AlphaArr0, MiuArr0, SigmaArr0

    
    #3、计算高斯分布函数
    def Gaussian_multi(self, x, miu, sigma):
        """
        多元高斯分布的密度函数
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            miu:该高斯分布的均值,1*d维
            sigma:该高斯分布的标准差,在此为d*d的协方差矩阵
        return:
            distributionArr:返回样本的概率分布1D数组
        """
        distributionArr = np.exp(-0.5*np.sum(np.multiply(np.dot(x-miu, np.linalg.pinv(sigma)), x-miu), axis=1))/\
        (np.power(2*np.pi, 0.5*self.d)*np.linalg.det(sigma)**0.5)
        return distributionArr

    #4、计算观测值y，高斯分布函数参数条件下，观测来自于第k个高斯分布的概率
    def Gama_Prob(self, x, AlphaArr, MiuArr, SigmaArr):
        """
        计算当观测值已知，是哪个高斯模型产品该观测值的概率
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            AlphaArr:每个高斯模型出现的先验概率,1*k维,k为聚类个数
            MiuArr:每个高斯模型的均值参数,k*d维
            SigmaArr:每个高斯模型的协方差矩阵参数,k*d*d维
        return:
            GamaProbArr:每个样本出现对应每个高斯模型分布概率的矩阵,m*k维
        """
        m = x.shape[0]
        GaussProbArr = np.zeros((m, self.k))
        for idx, value in enumerate(self.Dl_Yset):
            miu = MiuArr[idx]
            sigma = SigmaArr[idx]
            GaussProbArr[:,idx] = self.Gaussian_multi(x, miu, sigma)
        GamaProbArr = np.copy(np.multiply(GaussProbArr, AlphaArr))
        SumGamaProb = np.sum(GamaProbArr, axis=1).reshape(-1,1)
        return (GamaProbArr/SumGamaProb).round(4), GamaProbArr.round(4)
    
    #5、更新高斯分布函数参数
    def updateParas(self, x, GamaProbArr):
        """
        更新高斯分布函数的参数，包括均值和协方差矩阵
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            GamaProbArr:高斯分布函数的后验概率,m*k维
        return:
            newMiuArr:更新后的高斯分布函数的均值,k*d维
            newSigmaArr:更新后的高斯分布的协方差矩阵,k*d*d维
            newAlphaArr:更新后的高斯模型的混合系数,1*k维
        """
        m = x.shape[0]
        SumGamaProb = np.sum(GamaProbArr, axis=0)
        newMiuArr = np.zeros((self.k,self.d))
        newSigmaArr = np.zeros((self.k,self.d,self.d))
        for idx, value in enumerate(self.Dl_Yset):
            Gama = GamaProbArr[:,idx].reshape(-1,1)
            #更新均值
            newMiu = (np.sum(np.multiply(Gama, x), axis=0)+\
                      np.sum(self.Dl_Set[np.nonzero(self.Dl_Label==value)[0]], axis=0))/\
                      (SumGamaProb[idx]+self.l[idx])
            newMiuArr[idx] = newMiu
            #更新协方差矩阵
            newSigma = (np.dot(np.multiply(x-newMiu, Gama).T, x-newMiu)+\
                        np.dot((self.Dl_Set[np.nonzero(self.Dl_Label==value)[0]]-newMiu).T,(self.Dl_Set[np.nonzero(self.Dl_Label==value)[0]]-newMiu)))/\
                        (SumGamaProb[idx]+self.l[idx])
            newSigmaArr[idx] = newSigma
        #更新高斯模型混合系数
        newAlphaArr = (SumGamaProb.reshape(1,-1)+np.array(self.l).reshape(1,-1))/(m+sum(self.l))
        return newMiuArr, newSigmaArr, newAlphaArr

    #6、求似然函数值
    def calLLvalue(self, Dl_GaussProbArr, Du_GaussProbArr):
        Du_LLvalue = sum(np.log(Du_GaussProbArr.sum(axis=1)+1.0e-6))
        #设置一个0，1矩阵，用来判断第i个标签==第k个高斯模型
        Dl_yes = np.zeros((sum(self.l),self.k))
        for idx, value in enumerate(self.Dl_Yset):
            Dl_yes[np.nonzero(self.Dl_Label==value)[0],idx] = 1
        Dl_LLvalue = sum(np.log(np.multiply(Dl_GaussProbArr, Dl_yes).sum(axis=1)+1.0e-6))
        return Du_LLvalue+Dl_LLvalue
    
    #7、训练：判断是否符合停止条件
    def train(self, Dlx, Dly, Dux, iters):
        """
        循环迭代
        input:
            Dlx:已知标记的数据集
            Dly:已知标记的数据集的标签
            Dux:未知标记的数据集
            iters:迭代次数
        return:
            ClusterLabel:最终的聚类结果
        """
        #初始化参数
        self.Dl_Set = Dlx
        self.Dl_Label = Dly
        self.Du_Set = Dux
        AlphaArr0, MiuArr0, SigmaArr0 = self.initParas(Dlx, Dly)
        LLvalue0 = 0                #初始似然函数值
        LLvalueList = []            #最大似然值列表
        for i in range(iters):
            #计算已知标记数据集的高斯分布概率
            Dl_GamaProbArr, Dl_GaussProbArr = self.Gama_Prob(Dlx, AlphaArr0, MiuArr0, SigmaArr0)
            #计算未知标记数据集的高斯分布模型的后验概率，也就是已知观测下来自于第k个高斯分布函数的概率
            Du_GamaProbArr, Du_GaussProbArr = self.Gama_Prob(Dux, AlphaArr0, MiuArr0, SigmaArr0)
            #计算聚类结果
            ClusterLabel = np.argmax(Du_GamaProbArr, axis=1)
            #画分布图
            print("第%d次迭代："%i)
            #self.drawPics(Dux, MiuArr0, SigmaArr0, ClusterLabel)        #未标记数据
            #self.drawPics(Dlx, MiuArr0, SigmaArr0, Dly)                 #已标记数据
            Count_errs = sum(ClusterLabel[:40]!=0)+sum(ClusterLabel[40:82]!=1)+sum(ClusterLabel[82:]!=2)
            print("错误数：{}".format(Count_errs))
            #计算似然函数，并判断是否继续更新
            LLvalue1 = self.calLLvalue(Dl_GaussProbArr, Du_GaussProbArr)
            print('似然值：',LLvalue1)
            if len(LLvalueList) == 0:
                LLvalue0 = LLvalue1
            else:
                LLdelta = LLvalue1 - LLvalue0
                print('似然值提升：',LLdelta)
                if abs(LLdelta)<1.0e-6:
                    break
                else:
                    LLvalue0 = LLvalue1
            LLvalueList.append(LLvalue1)
            #继续迭代，更新函数参数
            MiuArr1, SigmaArr1, AlphaArr1 = self.updateParas(Dux, Du_GamaProbArr)
            MiuArr0 = np.copy(MiuArr1)
            SigmaArr0 = np.copy(SigmaArr1)
            AlphaArr0 = np.copy(AlphaArr1)
        self.ClusterLabel = ClusterLabel
        self.LL_ValueList = LLvalueList
        plt.plot(range(len(LLvalueList)), LLvalueList)
        plt.show()
        return

    #8-1、画图
    def drawPics(self, x, MiuArr, SigmaArr, Clusters):
        """
        画图，不同聚类类别的点分布，等高图
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            MiuArr, SigmaArr:高斯分布函数的参数
            Clusters:聚类结果
        out:
            散点图+等高分布图
        """
        plt.figure(figsize=(10,6))
        xgrid, ygrid, zgrid = self.calXYZ(x, MiuArr, SigmaArr)
        #c=plt.contour(xgrid,ygrid,zgrid,6,colors='black')
        plt.contour(xgrid,ygrid,zgrid,6,colors='black')
        plt.contourf(xgrid,ygrid,zgrid,6,cmap=plt.cm.Blues,alpha=0.5)
        #plt.clabel(c,inline=True,fontsize=10)
        for i in range(self.k):
            xi = x[Clusters==i,0]
            yi = x[Clusters==i,1]
            plt.scatter(xi, yi)
            plt.scatter(MiuArr[i,0], MiuArr[i,1], c='r', linewidths=5, marker='D')
        plt.show()
        return

    #8-2、计算网格状的高斯分布，用于画等高线
    def calXYZ(self, x, MiuArr, SigmaArr):
        """
        画等高图需要计算X,Y,Z
        input:
            x:样本集,m*d,其中m为样本数,d为样本维度数
            MiuArr, SigmaArr:高斯分布函数的参数
        return:
            xgrid:x的网格坐标
            ygrid:y的网格坐标
            zgrid:(x,y)网格坐标上高斯分布函数的概率
        """
        m = x.shape[0]
        x1 = np.copy(x[:,0])
        x1.sort()
        y1 = np.copy(x[:,1])
        y1.sort()
        x2,y2 = np.meshgrid(x1,y1)  # 获得网格坐标矩阵
        Gp = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                xi = x2[i,j]
                yi = y2[i,j]
                data = np.array([xi,yi])
                miuList=[]
                for miu, sigma in zip(MiuArr, SigmaArr):   
                    p = np.exp(-0.5*np.dot(np.dot((data-miu).reshape(1,-1), np.linalg.inv(sigma)), (data-miu).reshape(-1,1)))/\
                    np.power(2*np.pi, 0.5*self.d)*np.linalg.det(sigma)**0.5
                    miuList.append(p)
                Gp[i,j] = max(miuList)
        return x2, y2, Gp


##############鸢尾花数据，每类随机取10个样本为已知标记的样本，剩下的为未知标记样本
with open(r"D:\mywork\test\UCI_data\iris.data") as f:
    data = f.readlines()
trainSet = np.array([row.split(',') for row in data[:-1]])
trainSet = trainSet[:,:-1].astype('float')
#trainSet = trainSet[:,[1,3]]
trainSet = (trainSet - trainSet.mean(axis=0))/trainSet.std(axis=0)
labelSet = np.zeros(trainSet.shape[0])
labelSet[50:100] = 1; labelSet[100:] = 2                            #分段设置标签，0~50，50~100，100~150
Dl_index = np.random.randint(0, trainSet.shape[0], 30)              #随机选取的数据下标,有可能会重复
Dl_index = np.unique(Dl_index)                                      #对下标去重
DlSet = trainSet[Dl_index]                                          #已知标记的数据集Dl
DlLabel = labelSet[Dl_index]                                        #已知标记的数据标签Y
DuSet = np.delete(trainSet, Dl_index, axis=0)                       #作为未知标记的数据集Du
DuLabel = np.delete(labelSet, Dl_index, axis=0)                     #作为位置标记的数据标签Y

#训练
GMM = GaussianMix()
GMM.train(DlSet, DlLabel, DuSet, 50)
labels = GMM.ClusterLabel
Count_errs = sum(labels[:40]!=0)+sum(labels[40:82]!=1)+sum(labels[82:]!=2)






