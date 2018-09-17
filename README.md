# Machine-Learnning1
机器学习的深入学习
## 2018-8-29开始再次深入学习
主要利用了Numpy库，进一步了解了矢量化编程运算的意义，包括：矩阵内部元素的运算，矩阵与矩阵之间的运算，Linalg线性代数库等。
``` python
import numpy as np
from numpy import *
#np.linalg 线性代数库
linalg.det(M)   #矩阵行列式
linalg.inv(M)   #矩阵逆
linalg.matrix_rank(M)   #矩阵的秩
linalg.solve(Ma,Mre)    #Ma*Mb=Mre,求解Mb
```
## 2018-8-30各类距离的意义
``` python
dist = sqrt((Ma-Mb)*(Ma-Mb).T)   #欧氏距离
dist = sum(abs(Ma-Mb))    #曼哈顿距离
dist = abs(Ma-Mb).max()     #切比雪夫距离
dist = dot(Ma,Mb)/(linalg.norm(Ma)*linalg.norm(Mb)      #夹角余弦：考量两个向量方向上的差异
dist = nonzero(Ma-Mb)     #汉明距离：两个等长字符串中其中一个变为另一个所需要的最小替换次数
'''杰卡德相似系数是两个集合交集占并集的比例，可用来衡量相似度。'''
'''杰卡德距离是通过两个集合中不同元素占所有元素的比例，来衡量两个集合的区分度。'''
import scipy.spatial.distance as dist
dist.pdist(Ma,"jaccard")      #杰卡德距离
```
## 20108-8-31相关性
``` python
#相关系数
corrmatrix = np.corrcoef(M)     #出来的是二维相关系数矩阵，M为Ma和Mb相结合
corr = mean(multiply( (Ma-mean(Ma)),(Mb-mean(Mb)) ))/(std(Ma)*std(Mb))    #这里的自由度为N
#马氏距离
cov = mean(multiply( (Ma-mean(Ma)),(Mb-mean(Mb)) ))   #自由度为N的协方差
covmatrix = np.cov(Ma,Mb)     #自由度为(N-1)的二维协方差矩阵
covmatrix = np.cov(Mx)     #自由度为(N-1)的x维协方差矩阵
covinv = linalg.inv(covmatrix)    #协方差矩阵的逆矩阵，至于为什么要用逆矩阵估计是为了让每个样本点乘时，变量一一对应
Mt = M.T      #矩阵转置后，意义为多个样本，两列变量，这样在点乘时才可与协方差的对应上
DistMA = sqrt(dot   (dot ((Mt[0]-Mt[1]),covinv),(Mt[0]-Mt[1]).T )   )     #马氏距离：不管取0、1样本，1、2样本，或者0、2样本，距离都是一样。
```
## 2018-9-01特征值、标准化、可视化
``` python
#特征值和特征向量
A=[[8,1,6],[3,5,7],[4,9,2]]
evals, evecs = np.linalg.eig(A)     #evals为特征值，evecs为特征向量矩阵（每列为一个特征向量）
'''要是自己求的话，需满足lambda*Mx=A*Mx -> Mx(A-lambda)=0 -> |A-lambda|=0'''
equationA = [1,-15,-24,360]     #特征系数工程：x**3-15*x**2-24x+360=0
evals = np.roots(equationA)     #求根得出特征值
'''根据lambda*Mx=A*Mx，即可按照每个lanmbda特征值求出特征向量'''
#标准化——欧氏距离标准化
M=np.mat([[1,2,3],[4,5,6]])
std=np.std(M,axis=1)
'''或者'''
std=np.std(M.T,axis=0)
'''这一步求标准差没想通，为什么是样本求标准差，而不是变量向量求标准差，难道不是不同变量之间分别标准化嘛？'''
normM = (M-np.mean(M))/std      #标准化后的矩阵
delta = normM[0] - normM[1]
DistnormOU = np.sqrt(delta,delta.T)     #欧氏距离公式
```

# Machine-Learning2
文本分析
## 2018-09-10
``` python
'''
1、分词
2、分词结果转换为Bunch类并持久化
3、分词向量化（TF-IDF方法）——词频高且在词袋中出现的概率低，适合用来分类
4、向量结果转换为Bunch类并持久化
5、训练模型
6、导入测试集
7、模型评估
'''
```
## 2018-09-16 贝叶斯分类器
1、构造一个贝叶斯分类器
``` python
#定义训练集文本，简单构造一个。
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him','my'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
#编写贝叶斯算法类
class NBayes(object):
    def __init__(self):
        self.vocabulary = []    #文本词典
        self.idf = 0      #文本的idf权重
        self.tf = 0       #文本词向量矩阵
        self.tdm = 0      #P(x|y)每个类别的概率矩阵
        self.Pcates={}    #类别概率字典
        self.labels=[]     #文本分类列表
        self.doclength = 0  #文本数量
        self.vocablen = 0   #词典数量
        self.testset = 0    #测试集
    #导入和训练训练集数据
    def train_set(self,trainset,classVec):
        '''1、分类概率'''
        self.cate_prob(classVec)      #自建函数，计算每个分类类别的概率
        '''2、文本数'''
        self.doclength = len(self.trainset)     #训练集文本数
        '''3、词典'''
        tempset = set()       #创建词典集合
        [tempset.add(word) for doc in trainset for word in doc]     #不重复地合成每个分词
        self.vocabulary = list(tempset)     #转换为词典
        '''4、词典词长'''
        self.vocablen = len(self.vocabulary)
        '''5、统计词频'''
        self.wrd_freq(trainset)          #统计词频数据集，函数
        '''6、计算tdm'''
        self.build_tdm()                 #计算P(x|y)条件概率，函数
    #计算每个分类的概率
    def cate_prob(self,classVec):
        self.labels = classVec
        labelcates = set(self.labels)
        for templabel in labelcates:
            classtimes = self.labels.count(templabel)
            self.Pcates[templabel] = float(classtimes)/float(len(self.labels))
    #生成普通词频向量
    def wrd_freq(self,trainset):
        self.idf = np.zeros([1,self.vocablen])
        self.tf = np.zeros([self.doclength,self.vocablen])
        for row in range(self.doclength):
            for word in trainset[row]:
                self.tf[row,self.vocabulary.index(word)] += 1
            for singleword in set(trainset[row]):
                self.idf[0,self.vocabulary.index(singleword)] += 1
    #生成TF-IDF权重向量
    def wrd_tfidf(self,trainset):
        self.idf = np.zeros([1,self.vocablen])
        self.tf = np.zeros([self.doclength,self.vocablen])
        for row in range(self.doclength):
            for word in trainset[row]:
                self.tf[row,self.vocabulary.index(word)] += 1
            self.tf[row] = self.tf[row]/float(len(self.tf[row]))
            for singleword in set(trainset[row]):
                self.idf[0,self.vocabulary.index(singleword)] += 1
        self.idf = np.log(self.doclength/self.idf)
        self.tf = np.multiply(self.tf,self.idf)
    #生成tdm
    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates),self.vocablen])
        sumlist = np.zeros([len(self.Pcates),1])
        for row in range(self.doclength):
            self.tdm[self.labels[row]] += self.tf[row]
            sumlist = len(self.tdm[self.labels[rows]])
        self.tdm = self.tdm/sumlist
    #生成测试集向量
    def map2vocab(self,testdata):
        self.testset = np.zeros([1,self.vocablen])
        for word in testdata:
            self.testset[1,self.vocabulary.index(word)] += 1
    #预测函数
    def predict(self,testset):
        if self.testset.shape[1] != self.vocablen:
            print("测试集错误")
            exit(0)
        else:
            predvalue = 0
            predclass = ""
            for subtdm, subclass in zip(self.tdm, self.Pcates):
                temp = np.sum(testset*subtdm*self.Pcates[subclass])
                if temp > predvalue:
                    predvalue = temp
                    predclass = self.Pcates[subclass]
            return(predvalue,predclass)
#开始测试一个
dataset,listclass = loadDataSet()
nb = NBayes()
nb.train_set()
nb.map2vocab(dataset[0])
print(nb.predict(nb.testset))
```
2、直接调用scikit-learn实例(用scikit-learn直接训练简单的那个例子，看看tdm、tf、idf这些)
scikit-learn的TfidfVectorizer里面的tdm好像和实际的有点差别，需要再看一下。
