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
#杰卡德相似系数是两个集合交集占并集的比例，可用来衡量相似度。
#杰卡德距离是通过两个集合中不同元素占所有元素的比例，来衡量两个集合的区分度。
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
