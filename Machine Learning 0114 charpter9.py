# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:04:34 2018

@author: ecupl
"""

import numpy as np
import pandas as pd
import os,copy
import matplotlib.pyplot as plt
import cv2

#######################
#                     #
# OpenCV基本操作（一） #
#                     #
#######################
os.chdir(r"D:\mywork\test")
#======1、打开窗口======#
cv2.namedWindow("imgW",cv2.WINDOW_FREERATIO)
'''参数：
WINDOW_FREERATIO:自由拉缩窗口
WINDOW_KEEPRATIO:保持图片大小
WINDOW_NORMAL:根据窗口大小调节，保持关掉的窗口大小
WINDOW_AUTOSIZE:好像没变化
'''
#======2、读取图像======#
img1 = cv2.imread("images\\1.pgm",1)
#img2 = cv2.imread("images\\test2.jpg",0)
'''
1:打开彩色；0：打开灰色
如果不新建窗口，可直接打开，但默认WINDOW_KEEPRATIO
'''
#======3、在窗口展示图像======#
cv2.imshow("imgW",img1)
#cv2.imshow("imgW2",img2)
'''
可放置多个图像，（窗口名，已读取图像名）
'''
#======4、程序等待======#
cv2.waitKey(0)      
'''
按任意键执行下面程序，必须加上，不然未响应
'''
#======5、关闭窗口======#
#cv2.destroyWindow('imgW')
cv2.destroyAllWindows()

#######################
#                     #
# OpenCV基本操作（二） #
#                     #
#######################
WindowName = "imageWindow"
cv2.namedWindow(WindowName,cv2.WINDOW_NORMAL)
img = cv2.imread("images\\tooopen_sl_241494016392.jpg",0)
cv2.imshow(WindowName,img)
cv2.imwrite("images\\newpic.png",img,[int(cv2.IMWRITE_JPEG_QUALITY),50])
'''参数
默认保存图像质量为95，可自行设置
'''
cv2.waitKey(0)
cv2.destroyAllWindows()

#######################
#                     #
# OpenCV基本操作（三） #
#                     #
#######################
'''
matplotlib显示图
'''
img = cv2.imread("images\\test1.jpg",1)     #cv2读进来时BGR格式
img2 = img[:,:,[2,1,0]]                     #转换成Matplotlib中的RGB格式
plt.imshow(img2,cmap="gray",interpolation="bicubic")
plt.xticks([]);plt.yticks([])       #隐藏坐标轴
plt.show()
'''参数
interpolation:边界模糊度
cmap:如果X是3-D，则cmap会被忽略，而采用具体的RGB(A)值。
'''

#######################
#                     #
# OpenCV基本操作（四） #
#                     #
#######################
img = np.zeros((512,512,3))
#直线
cv2.line(img,(0,0),(510,510),(0,255,255),2)
'''起点，终点，颜色，像素'''
#矩形
cv2.rectangle(img,(10,0),(200,50),(255,0,0),-1)
'''左上角，右下角，颜色，像素'''
#圆形
cv2.circle(img,(300,300),60,(0,255,0),1)
'''圆心，半径，颜色，像素'''
#椭圆
cv2.ellipse(img,(400,400),(50,80),90,180,360,(0,0,255),-1)
'''中心，X边长和Y边长，旋转角度，开始角度，结束角度，颜色，像素'''
#多边形
pts =np.array([[50,50],[380,280],[150,500]])
cv2.polylines(img,[pts],1,(255,255,0),2)
'''顶点，是否闭环，颜色，像素'''
#文字
cv2.putText(img,"Test",(0,512),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)
'''内容，左下角，字样，大小，颜色，像素（粗细）'''
cv2.imshow("Draw",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#######################
#                     #
# Haar casade人脸识别  #
#                     #
#######################
from PIL import Image    #用来抓图的
import numpy as np
import pandas as pd
import os,copy
import matplotlib.pyplot as plt
import cv2

'''设置窗口'''
WindowName = "imageWindow"
cv2.namedWindow(WindowName,cv2.WINDOW_FREERATIO)
'''调用Haar分类器'''
face_casade = cv2.CascadeClassifier("D:\\python\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt_tree.xml")
'''读入图片'''
img = cv2.imread("images\\gyy1.jpg")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("images\\gyy1.jpg",0)
'''设置分类器参数'''
faces = face_casade.detectMultiScale(img2,scaleFactor=1.2,minNeighbors=2,minSize=(10,10),flags=cv2.CASCADE_SCALE_IMAGE)
'''参数
scaleFactor:区块大小，每次图像尺寸减小的比例
minNeighbors:检测个数
minSize:最小识别区块
maxSize:最大识别区块
flags:设置检测模式，如下：
        CASCADE_SCALE_IMAGE:按比例正常检测
        CASCADE_DO_CANNY_PRUNING:利用Canny边缘检测器来排除一些边缘很少或者很多的图像区域
        CASCADE_DO_ROUGH_SEARCH:只做初步检测
        CASCADE_FIND_BIGGEST_OBJECT:只检测最大的物体
'''
'''展示图片中的人脸检测结果'''
for x,y,w,h in faces:
    imgface = cv2.rectangle(img2,(x,y),(x+w,y+h),(255,255,255),2)
    #第一种抓图方式：直接抓取，反一下x，y
    roi_gray = img2[y:y+h,x:x+w]
    #第二种抓图方式，用pillow包
    roi_color = Image.open("images\\gyy1.jpg")
    roi_color2 = roi_color.crop((x,y,x+w,y+h))
cv2.imshow(WindowName,img2)
cv2.imshow("a",roi_gray)
roi_color2.save("colorface.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("face.jpg",img2)
 

#######################
#                     #
#  AdaBoost单层决策树  #
#                     #
#######################
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import metrics
os.chdir(r"D:\mywork\test\ML\郑捷《机器学习算法原理与编程实践》第2-9章节的源代码及数据集\adaboost")

'''1、封装函数'''
'''1-1 读入数据集'''
def loadDataSet(fileName):
    with open(fileName) as f:
        file = f.readlines()
    data = [[float(x) for x in row.split()] for row in file]
    data = np.array(data)
    train = data[:,:-1]
    label = data[:,-1]
    return train,label

'''1-2 AdaBoost训练'''
def AdaBoostTrain(dataSet,label,iters=50):
    weakClassSet = []           #弱分类器的容器
    m,n = np.shape(dataSet)
    '''1、初始化数据权重1/N，以及总体预测值'''
    D = np.ones((m,1))/float(m)
    aggValue = np.zeros((m,1))        
    for step in range(iters):
        '''2、通过单层树得到分类结果、分类误差'''
        bestFeat,bestprelabel,minerror = decisionTree(dataSet,label,D)
        '''3、根据最小误差得到弱分类器权重'''
        alpha = 0.5*np.log((1-minerror)/max(minerror,1e-16))
        '''4、更新弱分类器和数据集权重'''
        bestFeat['alpha'] = alpha
        weakClassSet.append(bestFeat)
        D = np.multiply(D,np.exp(-alpha*np.multiply(label,bestprelabel)))
        D = D/D.sum()
        '''5、最终错分个数和错分率，如果错分为0，则结束循环，否则继续'''
        aggValue += alpha*bestprelabel
        agglabel = np.sign(aggValue)
        errorNum = sum(agglabel!=label)[0];eRate = errorNum/m
        print(errorNum,eRate)
        if errorNum == 0:
            print("训练结束")
            break
    return weakClassSet,aggValue
    
'''1-3 单层决策树'''
def decisionTree(dataSet,label,D):
    m,n = np.shape(dataSet)
    bestFeat = dict()           #最优分隔点
    minerror = np.inf              #按权重的错误率
    bestprelabel = np.ones((m,1))   #最优预测分类
    steps = 10                  #迭代次数
    '''外循环：开始迭代，从变量开始'''
    for i in range(n):
        '''求列最大值、最小值、迭代步长'''
        columnMax = dataSet[:,i].max()
        columnMin = dataSet[:,i].min()
        stepLength = (columnMax-columnMin)/steps
        '''内循环：迭代每个步长'''
        for j in range(0,steps+1):
            threshold = columnMin+float(j)*stepLength       #阀值
            '''判断小于0的是0，还是大于0的是0'''
            for operator in ['lt','gt']:
                prelabel = splitDataSet(dataSet,i,threshold,operator)
                '''标出错误的数据集为1，错误率为权重相加'''
                errorSet = np.ones((m,1))
                errorSet[prelabel==label] = 0
                error = np.dot(D.T,errorSet)[0][0]
                if error < minerror:
                    minerror = error
                    bestprelabel = prelabel.copy()
                    bestFeat['column'] = i
                    bestFeat['threshold'] = threshold
                    bestFeat['operator'] = operator
    return bestFeat,bestprelabel,minerror

'''1-4 根据阈值分隔数据集'''
def splitDataSet(dataSet,column,threshold,operator):
    prelabel = np.ones((dataSet.shape[0],1))
    '''小于阈值的分类为0'''
    if operator=="lt":
        prelabel[dataSet[:,column]<=threshold] = -1
    
    else:       
        '''大于阈值的分类为0'''
        prelabel[dataSet[:,column]>=threshold] = -1
    return prelabel

'''1-5 预测分类'''
def classify(dataSet,weakClassSet):
    m,n=np.shape(dataSet)
    aggValue = np.zeros((m,1))
    for i in range(len(weakClassSet)):
        prelabel = splitDataSet(dataSet,weakClassSet[i]['column'],weakClassSet[i]['threshold'],weakClassSet[i]['operator'])
        aggValue += weakClassSet[i]['alpha']*prelabel
    return aggValue


'''2、正式开始训练'''
trainSet,trainLabel = loadDataSet("horseColicTraining.txt")
testSet,testLabel = loadDataSet("horseColicTest.txt")
'''训练弱分类器，先将Label转换为二维数组'''
trainLabel2D = trainLabel.reshape(-1,1)
trainLabel2D[trainLabel2D==0] = -1
weakClassSet,aggValue = AdaBoostTrain(trainSet,trainLabel2D,iters=10)       #10个弱分类器
'''对测试集进行预测，并将测试集数据转为一维数组'''
testaggValue = classify(testSet,weakClassSet)
pretestLabel = np.sign(testaggValue)
pretestLabel = pretestLabel.reshape(-1)
'''查看测试集的预测效果'''
testLabel[testLabel==0] = -1
error = sum(pretestLabel != testLabel);errorRate = error/len(testLabel)
print("错误:{}，错误率:{}".format(error,errorRate))
'''测试集结果：
错误:16，错误率:0.23880597014925373
'''

'''3、评价'''
'''ROC画图'''
'''第一种:书上的'''
cur = (1,1)
Pstep = 1/float(sum(testLabel==1));Nstep = 1/float(sum(testLabel==-1))
tpr=[];fpr=[]
aggValueIdx = np.argsort(testaggValue,axis=0)
labelarr=testLabel
plt.figure()
for idx in (aggValueIdx):
    if labelarr[idx] == 1:                #分错的，故真正例率要减掉
        delP = Pstep
        delN = 0
    else:                               #分对的，故假正例率要减掉
        delP = 0
        delN = Nstep
    plt.plot([cur[0],cur[0]-delN],[cur[1],cur[1]-delP])
    cur = (cur[0]-delN,cur[1]-delP)
    tpr.append(cur[1])
    fpr.append(cur[0])
plt.plot([0,1],[0,1],"r--")
plt.show()

'''第二种:自编的'''
trainLabel[trainLabel==-1] = 0
tpr = []; fpr = []; prs = []; thres = []
Pnum = float(sum(trainLabel==1))            #正样本数量          
Nnum = float(sum(trainLabel==0))           #负样本数量
aggValue1D = aggValue.reshape(-1)           #先转为一维数组
aggValueIdx = np.argsort(aggValue1D,axis=-1)       
for idx in aggValueIdx:
    threshold = aggValue1D[idx]     #先找到阈值
    prelabel = aggValue1D.copy()
    NegativeIdx = np.nonzero(prelabel<threshold)[0]
    PostiveIdx = np.nonzero(prelabel>=threshold)[0]
    prelabel[NegativeIdx]=0; prelabel[PostiveIdx]=1
    TN = sum((prelabel==trainLabel)*(trainLabel==0))
    TP = sum((prelabel==trainLabel)*(trainLabel==1))
    FP = sum((prelabel!=trainLabel)*(trainLabel==0))
    thres.append(threshold)
    tpr.append(TP/Pnum)
    fpr.append(1-(TN/Nnum))
    prs.append(TP/(TP+FP))
'''画图'''
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],"r--")
plt.show()
'''Area Under Curve'''
AUCnum = 0
PostiveIdx = np.nonzero(trainLabel==1)[0]
for idx in PostiveIdx:
    threshold = aggValue1D[idx]
    Num = sum((aggValue1D<=threshold)*(trainLabel==0))
    AUCnum += Num
AUC = AUCnum/(Pnum*Nnum)
AUC = metrics.roc_auc_score(trainLabel,aggValue1D)
print("AUC:{}".format(AUC))
AUC:0.8550468938620114


'''第三种:调用sklearn'''
'''
fpr:1-特异率
tpr:召回率
'''
fpr1, tpr1, thresholds=metrics.roc_curve(testLabel,testaggValue)
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],"r--")
plt.show()
AUC = metrics.roc_auc_score(testLabel,testaggValue)
AUC = metrics.auc(fpr,tpr)
'''AUC:0.8069148936170212'''

