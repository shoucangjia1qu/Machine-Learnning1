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

