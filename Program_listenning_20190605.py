# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:28:35 2019

@author: ecupl
"""

import os
import glob
import time
import shutil
import win32com
from win32com import client
import pandas as pd


#获取所有文件绝对路径
def searchFiles(path):
   files = glob.glob(os.path.join(path,'*.xls')) + glob.glob(os.path.join(path,'*.xlsx'))
   for f in files:
       '''过滤掉文件目录'''
       if os.path.isdir(f):
           continue
       else:
           yield f

#获取正确的文件
def getFiles(path)->list:
    filePath = []
    for p in searchFiles(path):
        filePath.append(p)
    return filePath

#获取新的文件路径
def newList(oldPath, newPath)->list:
    newList = []
    for subpath in newPath:
        if subpath not in oldPath:
            newList.append(subpath)
        else:
            pass
    return newList

#在监听目录下生成新的文件夹保存汇总结果
def mkdir(path, dirname:str):
    cwd = os.path.join(path,dirname)
    if not os.path.exists(cwd):
        os.mkdir(cwd)
    else:
        pass
    return cwd

#在excel中加入新的一行
def addExcel(path, excelList, columnIdx):
    global xlsApp, columnsSub
    '''生成汇总文件的文件夹'''
    sumDir = '汇总'
    sumPath = mkdir(path, sumDir)
    '''生成失败文件的文件夹'''
    failDir = '失败'
    failPath = mkdir(path,failDir)
    '''打开win32com模块'''
    xlsApp = win32com.client.DispatchEx('Excel.Application')                
    xlsApp.Visible = 1                      #显示文档        
    xlsApp.DisplayAlerts = False            #显示警告
    #判断是否有汇总文件，没有的话生成一个新的excel，有的话打开excel
    for excel in excelList:
        try:
            if not os.listdir(sumPath):
                excelNew = xlsApp.Workbooks.Add()               #打开新的excel，并选择第一个sheet
                objNew = excelNew.Worksheets(1)
                excelSub = xlsApp.Workbooks.Open(excel, UpdateLinks=False, ReadOnly=False, Format=None, Password='58880000')         #打开需要复制的excel，并选择第一个sheet
                objSub = excelSub.Worksheets(1)
                rowsSub = objSub.UsedRange.Rows.Count              #行数
                columnsSub = objSub.UsedRange.Columns.Count        #列数
                '''循环进行复制'''
                for row in range(1,rowsSub+1,1):
                    for column in range(1,columnsSub+3,1):
                        if (column == columnsSub+1) and row>1:
                            objNew.Cells(row,column).Value = excel          #加入文件路径
                        elif (column == columnsSub+2) and row>1:
                            timeStr = time.strftime('%Y-%b-%d %H:%M:%S',time.localtime())      #获取当地时间struct_time格式转换
                            objNew.Cells(row,column).Value = timeStr        #加入修改时间
                        else:
                            cellValue = objSub.Cells(row,column).Value      #需要复制的单元格的值
                            objNew.Cells(row,column).Value = cellValue      #复制值
                excelNew.SaveAs(sumPath+'\\汇总.xlsx')
                excelNew.Close()
                excelSub.Close()
                print(excel,'OK！')
            else:
                excelNew = xlsApp.Workbooks.Open(sumPath+'\\汇总.xlsx')
                objNew = excelNew.Worksheets(1)
                excelSub = xlsApp.Workbooks.Open(excel, UpdateLinks=False, ReadOnly=False, Format=None, Password='58880000')
                objSub = excelSub.Worksheets(1)
                rowsNew = objNew.UsedRange.Rows.Count               #汇总文件行数
                columnsNew = objNew.UsedRange.Columns.Count         #汇总文件列数
                rowsSub = objSub.UsedRange.Rows.Count               #需复制文件行数
                columnsSub = objSub.UsedRange.Columns.Count         #需复制文件列数
                '''循环进行复制'''
                for row in range(2,rowsSub+1,1):
                    for column in range(1,columnsSub+3,1):
                        if (column == columnsSub+1):
                            objNew.Cells(rowsNew+row-1,column).Value = excel          #加入文件路径
                        elif (column == columnsSub+2):
                            timeStr = time.strftime('%Y-%b-%d %H:%M:%S',time.localtime())      #获取当地时间struct_time格式转换
                            objNew.Cells(rowsNew+row-1,column).Value = timeStr        #加入修改时间
                        else:
                            cellValue = objSub.Cells(row,column).Value      #需要复制的单元格的值
                            objNew.Cells(rowsNew+row-1,column).Value = cellValue      #复制值
                excelNew.Save()
                excelSub.Close()
                print(excel,'OK！')
        except:
            failExcel = excel.split('\\')[-1]
            print('{}拷贝失败'.format(failExcel))
            shutil.copy(excel, failPath+'\\'+failExcel)
    try:
        excelNew.Close()
    except:
        pass
    addtimes(sumPath+'\\汇总.xlsx', columnIdx)
    xlsApp.Quit()
    return

#保存路径文件
def savePath(savePath, files:list, name:str):
    with open(savePath+'\\'+name, 'w') as ff:
        for i in files:
            ff.write(i+'\n')
    return

#读取路径文件
def loadPath(path, name:str):
    with open(path+'\\'+name, 'r') as f:
            Paths = f.readlines()
    oldPath = [i.strip() for i in Paths]
    return oldPath

#为记录打上次数标签
def addtimes(sumPathFile, columnIdx:int):
    data = pd.read_excel(sumPathFile, columns=0)
    m, n = data.shape
    if n == columnsSub + 2:
        n2 = n
    else:
        n2 = n-1
    cols = data.columns.tolist()
    columnName = cols[columnIdx-1]              #需要计数的列
    timeSeries = data[columnName].groupby(data[columnName]).count()       #每个样本的次数
    excelSum = xlsApp.Workbooks.Open(sumPathFile)           #再次打开汇总excel
    objSum = excelSum.Worksheets(1)
    for i in sorted(range(2, m+2, 1), reverse=True):
        idValue = objSum.Cells(i, columnIdx).Value
        timesValue = objSum.Cells(i, n2+1).Value
        if timesValue:
            break
        else:
            try:
                objSum.Cells(i, n2+1).Value = int(timeSeries[idValue])
#                print('OK!')
            except:
                print('please true ID')
    excelSum.Save()
    excelSum.Close()
    return

#train
def train(path, columnIdx):
    p = 'oldPath.txt'
    if not os.path.exists(path+'\\'+p):
        oldPath = []        #初始化文件路径为空列表
    else:
        oldPath = loadPath(path, p)     #加载旧路径文件
    while True:
        newPath = getFiles(path)        #获取当前路径下文件
        excelList = newList(oldPath, newPath)       #取出新的文件
        if excelList :
            addExcel(path, excelList, columnIdx)    #复制excel文件
            oldPath = newPath               #结束后，将新地址赋值给老地址
            savePath(path, oldPath, p)      #保存新路径文件
        print(time.strftime('%Y-%b-%d %H:%M:%S',time.localtime()))
        time.sleep(30)
        

if __name__ == '__main__':
    path = input(r"输入需要监听的目录:")
    columnIdx = int(input("请输入需要计数的列："))
    train(path, columnIdx)





