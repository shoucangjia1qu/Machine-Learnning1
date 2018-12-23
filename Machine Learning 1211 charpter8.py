# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 17:04:34 2018

@author: ecupl
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#######################
#                     #
#        SVM          #
#                     #
#######################
'''读取数据集'''
os.chdir(r"D:\mywork\test\ML")

def PlattSVM(object):
    def __init__():
        self.trainSet = 0
        self.Labels = 0
        
        
    
    '''读取数据集'''
    def loadData(self,filename):
        with open(filename,"r") as f:
            content = f.readlines()
            OriData = np.array([[float(comment) for comment in row.split()] for row in content])
        self.trainSet = OriData[:,:2]
        self.Labels = OriData[:,2]
            
            











