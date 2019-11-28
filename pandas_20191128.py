# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:59:44 2019

@author: ecupl
"""

import pandas as pd
import os

os.chdir("D:\\mywork\\test")

#Series.str.findall()函数
y = pd.Series(['ID:1 name:张三 age:24 income:13500',
               'ID:2 name:李四 age:27 income:25000',
               'ID:3 name:王二 age:21 income:8000'])
##通过指定字符进行匹配
y.str.findall("24")
y.str.findall("name")
##通过正则进行匹配
y.str.findall('age:(\d+)')
y.str.findall('\d+')
##将匹配上的字符转换为其他数据类型
y.str.findall('\d+').str[1].astype(int)



