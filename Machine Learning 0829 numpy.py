# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:27:40 2018

@author: ecupl
"""

import numpy as np
#生成1行3列普通矩阵
matrix0 = np.mat([1,2,3])
#生成全0或全1矩阵
myzero = np.zeros([3,5])
myone = np.ones([5,3])
#生成两个3x3矩阵
matrix1 = np.mat([[1,2,3],[4,5,6],[10,7,8]])
matrix2 = np.mat([[1,2,3],[3,2,1],[10,20,30]])
###################矩阵元素运算##################
#矩阵元素相加
print(matrix1+matrix2)
#矩阵数乘
print(0*matrix1)
#矩阵元素求和
np.sum(matrix1)
matrix1.sum(axis=1)     #列求和
matrix1.sum(axis=0)     #行求和
#矩阵元素求积
np.multiply(matrix1,matrix2)
#矩阵元素求幂
np.power(matrix1,2)
###################矩阵运算##################
#矩阵相乘
matrix1*matrix2
matrix1.dot(matrix2)
#矩阵转置
matrix1.T
#矩阵行列数
matrix1.shape
np.shape(matrix1)
#按行切片
matrix1[0]
#按列切片
matrix1.T[0]
#复制矩阵
matrix1.copy()
#矩阵元素比较
matrix1<matrix2
##################Linalg线性代数库###################
#矩阵的行列式
np.linalg.det(matrix1)
#矩阵的逆
np.linalg.inv(matrix1)
np.dot(matrix1,np.linalg.inv(matrix1))      #单位矩阵
#矩阵的秩
np.linalg.matrix_rank(matrix1)              #3
np.linalg.matrix_rank(matrix2)              #2
np.linalg.matrix_rank(myone)                #1
#可逆矩阵求解
np.linalg.solve(matrix1,np.mat([[1,0,1],[2,2,2]]).T)
##################Linalg距离###################
#设置两点坐标
A=np.array((8,1,6))         #3*1
A=np.array([8,1,6])         #3*1
A=np.array([[8,1,6]])       #1*3
np.linalg.norm(A)           #点A到原点（0，0，0）的剧烈
np.sqrt(np.sum(np.power(A,2)))      #平方和开根号
B=np.array([[4,2,5]])
np.linalg.norm(A-B)         #A和B两点之间的距离
np.sqrt(np.sum(np.power((A-B),2)))      #平方和开根号

##################0830各类距离的意义以及python的实现###################
##闵科夫斯基距离（Minkowski Distance）
##欧氏距离（Euclidean Distance）
Ma=np.mat([1,2,3])
Mb=np.mat([4,7,5])
dis = np.sqrt((a-b)*((a-b).T))              #其实就是L2范数
dis = np.linalg.norm(a-b)
dis = np.sqrt(np.sum(np.power((a-b),2)))
##曼哈顿距离（Manhattan Distance）
dis = np.sum(abs(a-b))                      #其实就是L1范数
##切比雪夫距离（Chebyshev Distance）
dis = abs(a-b).max()                        #坐标差值最大的值
##夹角余弦（Cosine）
dis = np.dot(a,b.T)/(np.linalg.norm(a)*np.linalg.norm(b))
##汉明距离(Hamming Distance)
a=np.mat([1,1,0,1,0,1,0,0,1])
b=np.mat([0,1,1,0,0,0,1,1,1])
smstr = np.nonzero(a-b)
print(np.shape(smstr)[1])               #信息编码转成另外一个所需要的最小替换次数
##杰卡德相似系数（Jaccard Similarity Coefficient）
###杰卡德相似系数是指两个集合中的交集元素占并集元素的比重
###杰卡德距离是指两个集合的非交集元素占并集元素的比重
import scipy.spatial.distance as dist
M = np.mat([[1,1,0,1,0,1,0,0,1],[0,1,1,0,0,0,1,1,1]])
dist.pdist(M,metric="jaccard")

##################0831相关性###################
#相关系数，衡量两个特征列之间的相关度
cov = np.mean(np.multiply((Ma-np.mean(Ma)),(Mb-np.mean(Mb))))       #协方差，不过这里的自由度是n，不是(n-1)
corr1 = cov/(np.std(Ma)*np.std(Mb))         #相关系数
corr2 = np.corrcoef(Ma,Mb)          #直接得出相关系数矩阵
##################马氏距离###################
#马氏距离排除变量之间的干扰，比如：身高、体重单位不同的影响
#协方差矩阵
covmatrix = np.cov(Ma,Mb)
covinv = np.linalg.inv(covmatrix)       #协方差矩阵的逆矩阵，两个变量之间的协方差
M = np.mat([[1,2,3],[4,7,5]])
Mt = M.T                #相当于三个样本，两个变量
distma = np.sqrt(np.dot(np.dot((Mt[0]-Mt[1]),covinv),(Mt[0]-Mt[1]).T))      #样本0和样本1的马氏距离
distma = np.sqrt(np.dot(np.dot((Mt[1]-Mt[2]),covinv),(Mt[1]-Mt[2]).T))      #样本1和样本2的马氏距离，结果和上面一样
distma = np.sqrt(np.dot(np.dot((Mt[0]-Mt[2]),covinv),(Mt[0]-Mt[2]).T))      #样本1和样本2的马氏距离，结果也和上面一样





















