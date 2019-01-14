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


