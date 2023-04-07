#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 21:53:03 2023

@author: jane_hsieh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math
import os

#Set the current working directory / or manually type-in
cwdir = os.getcwd() #'/Users/jane_hsieh/Library/CloudStorage/OneDrive-國立陽明交通大學/Data Science Analysis Templates/Anomaly Detection/Subspace-based Anomaly Detection/PCA-based Anomaly Detection'
#or manually type-in
cwdir = "/Users/jane_hsieh/Library/CloudStorage/OneDrive-國立陽明交通大學/Data Science Analysis Templates/Anomaly Detection/Subspace-based Anomaly Detection/PCA-based Anomaly Detection"
os.chdir(cwdir)

#indicate the directory path for the fold where 4 MNIST data are located (default: in cwdir)
data_dir = './MNIST Database'
    
    
#===================================== 0. Input Data: MNIST (from web) ======================================
#1.1 Input training/testing data and their labels (X, y)
from mlxtend.data import loadlocal_mnist #make sure installed via conda
Train_X, Train_y = loadlocal_mnist(
        images_path=data_dir+'/train-images-idx3-ubyte', 
        labels_path=data_dir+'/train-labels-idx1-ubyte')

Test_X, Test_y = loadlocal_mnist(
        images_path=data_dir+'/t10k-images-idx3-ubyte', 
        labels_path=data_dir+'/t10k-labels-idx1-ubyte')

## Transform into DataFrame form
Train_X = pd.DataFrame(Train_X)
Train_y = pd.DataFrame(Train_y, columns = ["digit"])
#Test_X = pd.DataFrame(Test_X)
#Test_y = pd.DataFrame(Test_y, columns = ["digit"])

'''
## Inspenct the frequency of each class (digit 0-9)
unique_elements, counts_elements = np.unique(Train_y, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


Train_N = Train_X.shape[0]
print("Percentage of each class of Train_y data: \n {}".format(np.asarray((unique_elements, counts_elements/Train_N))))
'''
p = Train_X.shape[1]
width = int(math.sqrt(p))
shape =  (width, width)