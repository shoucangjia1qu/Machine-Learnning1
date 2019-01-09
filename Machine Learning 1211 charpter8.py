# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:04:34 2018

@author: ecupl
"""

import numpy as np
import pandas as pd
import os,copy
import matplotlib.pyplot as plt

#######################
#                     #
#        SVM          #
#                     #
#######################
os.chdir(r"D:\mywork\test\ML")

import copy
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
        
        
    
    '''读取数据集'''
    def loadData(self,filename):
        with open(filename,"r") as f:
            content = f.readlines()
            OriData = np.array([[float(comment) for comment in row.split()] for row in content])
        self.trainSet = OriData[:,:2]
        self.Labels = OriData[:,2].reshape((len(OriData),1))
    
    '''初始化'''        
    def initparam(self):
        m,n = np.shape(self.trainSet)
        self.alpha = np.zeros((m,1))
        self.eCache = np.zeros((m,2))          #一列为标注误差是否更新，一列是记载的误差
        self.K = np.zeros((m,m))
        for k in range(m):
            self.K[k,:] = self.kernels(self.trainSet,self.trainSet[k,:])
    
    '''构造核函数'''
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

    '''计算误差函数'''
    def calEk(self,i):
        Ek = float(np.dot(np.multiply(self.alpha,self.Labels).T,self.K[:,i])) + self.b - self.Labels[i,0]      # Yp=W*X+b;E=Yp-Y
        return Ek
    
    '''选择子循环的变量'''
    def chooseJ(self,i,Ei):
        maxj = -1               #取0的话容易变成0.13
        maxEj = 0
        maxDeltaE = 0
        self.eCache[i] = (1,Ei)
        SvmError = np.nonzero(self.eCache[:,0])[0]         #判断当前误差的个数，如果只有一个说明是第一次，就可以随机选取J
        if len(SvmError)>1:
            '''目前不止一个误差'''
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
            '''目前只有一个误差'''
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
    
    '''主函数：主循环'''
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
                        '''(alpha1<self.C and y1*Ei<-self.tol) or (alpha1>0 and y1*Ei>self.tol)'''
                        '''判定是否符合KKT条件，不符合的就进行内循环'''
                        AlphaChange += self.inner(i,Ei,y1)        #内循环返还标识
                print("KKT",step,AlphaChange)
                step+=1
            if flag : flag = False                                                                                          #转换标志位：切换到另一种
            elif (AlphaChange == 0) :flag = True                            #改变主循环数据集为（0，C）
        self.svIdx = np.nonzero(self.alpha>0)[0]            #支持向量的下标
        self.sptVects = self.trainSet[self.svIdx]           #支持向量
        self.SVlabels = self.Labels[self.svIdx]             #支持向量的标签
        print(step)

    '''主函数：内循环'''
    def inner(self,i,Ei,y1):
        j,Ej = self.chooseJ(i,Ei)           #生成第二个变量
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
        '''求解新的alpha变量'''
        eta = self.K[i,i] + self.K[j,j] - 2*self.K[i,j]
        if eta<=0:
            return 0
        '''未剪枝的newAlpha2'''
        self.alpha[j,0] = oldAlpha2 +y2*(Ei-Ej)/eta
        '''选定最终的Alpha2'''
        self.alpha[j,0] = self.cutalpha(self.alpha[j,0],L,H)
        '''计算最终的Alpha1'''
        if abs(oldAlpha2-self.alpha[j,0])<0.00001:
            return 0
        self.alpha[i,0] = oldAlpha1 + y1*y2*(oldAlpha2-self.alpha[j,0])
        '''计算最终的误差Ei和Ej'''
        self.eCache[j] = (1,self.calEk(j))
        self.eCache[i] = (1,self.calEk(i))
        '''计算最终的b'''
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
        for i in range(shape(self.trainSet)[0]) :
            if self.alpha[i] != 0 :
                ax.scatter(self.trainSet[i, 0], self.trainSet[i, 1], c = 'green', marker = 's')
            elif self.Labels[i] == 1 :
                ax.scatter(self.trainSet[i, 0], self.trainSet[i, 1], c = 'blue', marker = 'o')
            elif self.Labels[i] == -1 :
                ax.scatter(self.trainSet[i, 0], self.trainSet[i, 1], c = 'red', marker = 'o')

    
    '''预测'''
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
svm.scatterplot(plt)
plt.show()
print(svm.classify(svm.trainSet, svm.Labels))
print(svm.svIdx)
print(shape(svm.sptVects)[0])
print("b: ", svm.b)

'''[  3   9  26  27  33  40  49  52  53  59  79  81 100 101 102 103 104 106
 107 109 130 133 148 198 199]
25
b:  [[ 8.54969697]]'''


#######################
#                     #
#      文本分类        #
#                     #
#######################
import pickle
from sklearn.svm import LinearSVC    #导入线性SVM
'''1、导入数据'''
with open("D:\\mywork\\test\\ML_Chinese\\tfidfspace.dat","rb") as f1:
    train = pickle.load(f1)
with open("D:\\mywork\\test\\ML_Chinese\\test_set.dat","rb") as f2:
    test = pickle.load(f2)
'''2、构建测试集tdm向量'''
from sklearn.datasets.base import Bunch
tfidftest = Bunch(target_name=test.target_name, label=test.label, filenames=test.filenames,
                   tdm=[], vocabulary={})
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer    #TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer     #TF-IDF向量生成类
stoplist = readfile(stop_path).splitlines()         #见第二章的函数
'''2-1 构建测试集向量时需使用训练集词袋向量'''
vectorizer = TfidfVectorizer(stop_words=stoplist, sublinear_tf=True, max_df=0.5,
                             vocabulary=train.vocabulary)
transformer = TfidfTransformer()    #统计每个词语的TF-IDF权重
text=[i.decode("GBK","ignore") for i in test.contents]     #将二进制转为unicode
tfidftest.tdm = vectorizer.fit_transform(text)
tfidftest.vocabulary = train.vocabulary
'''3、建模'''
svm = LinearSVC(penalty='l2',dual=False,tol=0.0001)
svm.fit(train.tdm,train.label)
pre=svm.predict(tfidftest.tdm)
true = tfidftest.label
'''4、评估'''
from sklearn import metrics
print(metrics.classification_report(true,pre))
error = 0
for i in range(len(pre)):
    if pre[i] != true[i]:
        error+=1
AUC = (len(pre)-error)/len(pre)
print("AUC:%.2f"%AUC)

'''AUC:0.90'''

#######################
#                     #
#      参数解释        #
#                     #
#######################
'''1、LinearSVC
penalty:正则化参数，L1和L2两种参数可选，仅LinearSVC有。
loss:损失函数，有‘hinge’和‘squared_hinge’两种可选，前者又称L1损失，后者称为L2损失，默认是是’squared_hinge’，其中hinge是SVM的标准损失，squared_hinge是hinge的平方。
dual:是否转化为对偶问题求解，默认是True。
tol:残差收敛条件，默认是0.0001，与LR中的一致。
C:惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。
multi_class:负责多分类问题中分类策略制定，有‘ovr’和‘crammer_singer’ 两种参数值可选，默认值是’ovr’，'ovr'的分类原则是将待分类中的某一类当作正类，其他全部归为负类，通过这样求取得到每个类别作为正类时的正确率，取正确率最高的那个类别为正类；‘crammer_singer’ 是直接针对目标函数设置多个参数值，最后进行优化，得到不同类别的参数值大小。
fit_intercept:是否计算截距，与LR模型中的意思一致。
class_weight:与其他模型中参数含义一样，也是用来处理不平衡样本数据的，可以直接以字典的形式指定不同类别的权重，也可以使用balanced参数值。
verbose:是否冗余，默认是False.
random_state:随机种子的大小。
max_iter:最大迭代次数，默认是1000。
'''

'''2、SVC
C: 惩罚系数，用来控制损失函数的惩罚系数，类似于LR中的正则化系数。C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样会出现训练集测试时准确率很高，但泛化能力弱，容易导致过拟合。 C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强，但也可能欠拟合。
kernel: 算法中采用的和函数类型，核函数是用来将非线性问题转化为线性问题的一种方法。参数选择有RBF, Linear, Poly, Sigmoid，precomputed或者自定义一个核函数, 默认的是"RBF"，即径向基核，也就是高斯核函数；而Linear指的是线性核函数，Poly指的是多项式核，Sigmoid指的是双曲正切函数tanh核；。
degree: 当指定kernel为'poly'时，表示选择的多项式的最高次数，默认为三次多项式；若指定kernel不是'poly'，则忽略，即该参数只对'poly'有用。（多项式核函数是将低维的输入空间映射到高维的特征空间）
gamma: 核函数系数，该参数是rbf，poly和sigmoid的内核系数；默认是'auto'，那么将会使用特征位数的倒数，即1 / n_features。（即核函数的带宽，超圆的半径）。gamma越大，σ越小，使得高斯分布又高又瘦，造成模型只能作用于支持向量附近，可能导致过拟合；反之，gamma越小，σ越大，高斯分布会过于平滑，在训练集上分类效果不佳，可能导致欠拟合。 
coef0: 核函数常数值(y=kx+b中的b值)，只有‘poly’和‘sigmoid’核函数有，默认值是0。
shrinking :  是否进行启发式。如果能预知哪些变量对应着支持向量，则只要在这些样本上训练就够了，其他样本可不予考虑，这不影响训练结果，但降低了问题的规模并有助于迅速求解。进一步，如果能预知哪些变量在边界上(即a=C)，则这些变量可保持不动，只对其他变量进行优化，从而使问题的规模更小，训练时间大大降低。这就是Shrinking技术。 Shrinking技术基于这样一个事实：支持向量只占训练样本的少部分，并且大多数支持向量的拉格朗日乘子等于C。
probability: 是否使用概率估计，默认是False。必须在 fit( ) 方法前使用，该方法的使用会降低运算速度。
tol: 残差收敛条件，默认是0.0001，即容忍1000分类里出现一个错误，与LR中的一致；误差项达到指定值时则停止训练。
cache_size: 缓冲大小，用来限制计算量大小，默认是200M。
class_weight :  {dict, ‘balanced’}，字典类型或者'balance'字符串。权重设置，正类和反类的样本数量是不一样的，这里就会出现类别不平衡问题，该参数就是指每个类所占据的权重，默认为1，即默认正类样本数量和反类一样多，也可以用一个字典dict指定每个类的权值，或者选择默认的参数balanced，指按照每个类中样本数量的比例自动分配权值。如果不设置，则默认所有类权重值相同，以字典形式传入。 将类i 的参数C设置为SVC的class_weight[i]*C。如果没有给出，所有类的weight 为1。'balanced'模式使用y 值自动调整权重，调整方式是与输入数据中类频率成反比。如n_samples / (n_classes * np.bincount(y))。（给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C。如果给定参数'balance'，则使用y的值自动调整与输入数据中的类频率成反比的权重。）
verbose :  是否启用详细输出。在训练数据完成之后，会把训练的详细信息全部输出打印出来，可以看到训练了多少步，训练的目标值是多少；但是在多线程环境下，由于多个线程会导致线程变量通信有困难，因此verbose选项的值就是出错，所以多线程下不要使用该参数。
max_iter: 最大迭代次数，默认是-1，即没有限制。这个是硬限制，它的优先级要高于tol参数，不论训练的标准和精度达到要求没有，都要停止训练。
decision_function_shape ：  原始的SVM只适用于二分类问题，如果要将其扩展到多类分类，就要采取一定的融合策略，这里提供了三种选择。‘ovo’ 一对一，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果，决策所使用的返回的是（样本数，类别数*(类别数-1)/2）； ‘ovr’ 一对多，为one v rest，即一个类别与其他类别进行划分，返回的是(样本数，类别数)，或者None，就是不采用任何融合策略。默认是ovr，因为此种效果要比oro略好一点。
random_state: 在使用SVM训练数据时，要先将训练数据打乱顺序，用来提高分类精度，这里就用到了伪随机序列。如果该参数给定的是一个整数，则该整数就是伪随机序列的种子值；如果给定的就是一个随机实例，则采用给定的随机实例来进行打乱处理；如果啥都没给，则采用默认的 np.random实例来处理。 
'''

'''3、NuSVC
nu： 训练误差部分的上限和支持向量部分的下限，取值在（0，1）之间，默认是0.5
'''


















