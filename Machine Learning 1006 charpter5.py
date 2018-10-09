# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:27:40 2018

@author: ecupl
"""

###################逐次逼近法/迭代法求解##################
import numpy as np
import matplotlib.pyplot as plt
'''消元法'''
A = np.mat([[8,-3,2],[4,11,-1],[6,3,12]])
b = np.mat([[20,33,36]])
result = np.linalg.solve(A,b.T)
print(result)
'''迭代法'''
error = 1.0e-6
steps = 100
n=3
B0=np.mat([[0,3/8,-2/8],[-4/11,0,1/11],[-6/12,-3/12,0]])
f=np.mat([[20/8,33/11,36/12]])
xk = np.zeros((n,1))
errorlist=[]
for k in range(steps):
    xk_1 = xk
    xk = B0*xk + f.T
    errorlist.append(np.linalg.norm(xk-xk_1))
    print('第{}次迭代,解是{},误差{}'.format(k+1,xk,errorlist[-1]))
    print('-------------------------------')
    if errorlist[-1] < error:
        break
#画图    
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure()
x = np.linspace(1,k+1,k+1)
y = errorlist
plt.scatter(x,y,c='r',marker='^',linewidths=5)
plt.xlabel('迭代次数')
plt.ylabel('误差')
plt.show()

###################线性感知器##################
x1=np.linspace(0,9,10)
x3=3*x1+2+np.random.rand(10)*0.2
data1=np.array((x1,x3)).T
y1=np.linspace(10,19,10)
y3=3*y1+2-np.random.rand(10)*0.2
data2=np.array((y1,y3)).T
a=data1.tolist()
b=data2.tolist()
a.extend(b)
trainSet = np.array(a)
Bi = np.ones((20,1))
train = np.column_stack((Bi,trainSet))          #生成训练集数据
label = np.row_stack((np.ones((10,1)),-np.ones((10,1))))            #生成训练集标签
'''创建单层感知器'''
X = np.array([[1,3,3],
              [1,4,3],
              [1,1,1]])
#标签
Y = np.array([[1],
              [1],
              [-1]])

W0 = np.zeros((3,1))
alpha = 0.11
steps = 500
for i in range(steps):
    gradient = np.dot(X,W0)     #计算梯度
    O = np.sign(gradient)
    Wt = W0 + alpha*np.dot(X.T,(Y-O))       #计算误差
    W0 = Wt

###################线性感知器##################
import numpy as np
import matplotlib.pyplot as plt

'''准备数据集'''
with open(r"D:\mywork\test\ML\dataSet_BP.txt",'r') as f:
    content=f.readlines()
trainList = [row.split() for row in content]
x,y=np.shape(trainList)
trainSet=np.zeros((x,y))    #初始化矩阵
for i in range(x):
    for j in range(y):
        trainSet[i][j] = float(trainList[i][j])
train = np.array([i for i in trainSet if i[0]<=6])
'''画图'''
xdata1 = [i[0] for i in trainSet if i[2]==0 and i[0]<=6]
ydata1 = [i[1] for i in trainSet if i[2]==0 and i[0]<=6]
xdata2 = [i[0] for i in trainSet if i[2]==1 and i[0]<=6]
ydata2 = [i[1] for i in trainSet if i[2]==1 and i[0]<=6]
plt.figure()
plt.scatter(xdata1,ydata1,c='r',marker='^')
plt.scatter(xdata2,ydata2,c='b',marker='o')
plt.show()
'''构造线性感知器'''
target = train[:,-1].reshape((158,1))       #标签
trainData = np.column_stack((np.ones((158,1)),train[:,:2]))
alpha = 0.001
steps = 500
W = np.ones((3,1))
for k in range(steps):
    gradient = np.dot(trainData,W)
    output = 1/(1+np.exp(-gradient))
    error = target - output
    W = W + alpha*np.dot(trainData.T,error)
    #画图展示变化
    Xre = np.linspace(-6,6,100)
    Yre = -(W[0]+Xre*W[1])/W[2]
    xdata1 = [i[0] for i in trainSet if i[2]==0 and i[0]<=6]
    ydata1 = [i[1] for i in trainSet if i[2]==0 and i[0]<=6]
    xdata2 = [i[0] for i in trainSet if i[2]==1 and i[0]<=6]
    ydata2 = [i[1] for i in trainSet if i[2]==1 and i[0]<=6]
    plt.figure()
    plt.scatter(xdata1,ydata1,c='r',marker='^')
    plt.scatter(xdata2,ydata2,c='b',marker='o')
    plt.plot(Xre,Yre)
    plt.show()
'''对测试集进行分类'''
testData = np.array([[1,-4,5],[1,2,7],[1,3,-1],[1,2,0.5],[1,0,0]])
Re = 1/(1+np.exp(-(np.dot(testData,W))))
#测试集画图
plt.figure()
plt.scatter(xdata1,ydata1,c='r',marker='^')
plt.scatter(xdata2,ydata2,c='b',marker='o')
plt.plot(Xre,Yre)
plt.scatter(testData[:,1],testData[:,2],c='g',marker='o',linewidths=10)
plt.show()

###################算法分析##################
'''评估算法的近似结果'''
'''
1、超平面分析
2、斜率和截距分析：初始值的震荡
3、权重收敛评估
4、算法的总体评价
'''
'''==========超平面的变化趋势=========='''
Wlist = []          #初始化权重列表，用来保存权重
alpha = 0.001
steps = 1000
W = np.ones((3,1))
for k in range(steps):
    gradient = np.dot(trainData,W)
    output = 1/(1+np.exp(-gradient))
    error = target - output
    W = W + alpha*np.dot(trainData.T,error)
    Wlist.append(W)     #保存当前的权重

#画图
plt.figure()
plt.scatter(xdata1,ydata1,c='r',marker='^')
plt.scatter(xdata2,ydata2,c='b',marker='o')
Xre = np.linspace(-6,6,100)
for index in range(len(Wlist)):
    if index%20 == 0:
        Yre = -(Wlist[index][0]+Xre*Wlist[index][1])/Wlist[index][2]
        plt.plot(Xre,Yre)
        plt.annotate("hplane:"+str(index),xy=(Xre[99],Yre[99]))
plt.show()
'''==========超平面的收敛估计：截距的变化=========='''
plt.figure()
Xb=np.linspace(0,999,1000)
Yb=[-i[0]/i[2] for i in Wlist]
M = 50
plt.subplot(211)
plt.plot(Xb[:M],Yb[:M])
plt.subplot(212)
plt.plot(Xb[M:],Yb[M:],c='r')
plt.show()
'''==========超平面的收敛估计：斜率的变化=========='''
plt.figure()
Xa=np.linspace(0,999,1000)
Ya=[-i[1]/i[2] for i in Wlist]
M = 20
plt.subplot(211)
plt.plot(Xa[:M],Ya[:M])
plt.subplot(212)
plt.plot(Xa[M:],Ya[M:],c='r')
plt.show()
'''==========权重向量收敛估计=========='''
plt.figure()
Xw=np.linspace(0,999,1000)
Yw0=[i[0] for i in Wlist]
Yw1=[i[1] for i in Wlist]
Yw2=[i[2] for i in Wlist]
plt.subplot(3,1,1)
plt.plot(Xw,Yw0)
plt.subplot(3,1,2)
plt.plot(Xw,Yw1,c='r')
plt.subplot(3,1,3)
plt.plot(Xw,Yw2,c='g')
plt.show()
'''再加500次迭代后收敛效果更好'''

###################随机梯度下降法：算法改进与评估##################

