#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:59:27 2019

@author: jane_hsieh

Topic: NCTU_MLSP (MNIST data) Use spectral clustering method for dimention reduction (extract features)



Supplement: Install MNIST data
    1. Data source: http://yann.lecun.com/exdb/mnist/
    2. Tutorial of input:
        (1. Python code: http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/
        (2. Prerequisite of package "mlxtend" to download: http://rasbt.github.io/mlxtend/installation/#conda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math

"""
Steps:
    0. Input MNIST dataset
    1. Since the dimension of the feature data is too high, I hence do spectral clustering for dimension reduction, to save times
        1.1 Calculate affinity matrix A: s(x, y)= rbf_kernel similarity if class(x)=class(y); o.w. s(x, y)= 0; and double scale A => L
        1.2. Feature extraction via spectral dimension reduction: X (dim = Train_N x p) = > W (dim = Train_N x k)
    2. Do the same spectral dimension reduction for Test data
"""

#===================================== 0. Input Data: MNIST (from web) ======================================
#1.1 Input training/testing data and their labels (X, y)
from mlxtend.data import loadlocal_mnist #make sure installed via conda

#Please indicate the directory path for the fold where 4 MNIST data are located
data_dir = '/Users/jane_hsieh/Library/CloudStorage/OneDrive-國立陽明交通大學/Data Science Analysis Templates/Datasets/MNIST Database'

Train_X, Train_y = loadlocal_mnist(
        images_path=data_dir+'/train-images-idx3-ubyte', 
        labels_path=data_dir+'/train-labels-idx1-ubyte')

Test_X, Test_y = loadlocal_mnist(
        images_path=data_dir+'/t10k-images-idx3-ubyte', 
        labels_path=data_dir+'/t10k-labels-idx1-ubyte')

## Transform into DataFrame form
Train_X = pd.DataFrame(Train_X)
Train_y = pd.DataFrame(Train_y, columns = ["digit"])
Test_X = pd.DataFrame(Test_X)
Test_y = pd.DataFrame(Test_y, columns = ["digit"])

'''
## Inspenct the frequency of each class (digit 0-9)
unique_elements, counts_elements = np.unique(Train_y, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

cumcounts_elements = counts_elements.cumsum()
Train_N = Train_X.shape[0]
print("Percentage of each class of Train_y data: \n {}".format(np.asarray((unique_elements, counts_elements/Train_N))))

"""
Frequency of unique values of the said array:
[[   0    1    2    3    4    5    6    7    8    9]
 [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]]
Percentage of each class of Train_y data: 
 [[0.         1.         2.         3.         4.         5.
  6.         7.         8.         9.        ]
 [0.09871667 0.11236667 0.0993     0.10218333 0.09736667 0.09035
  0.09863333 0.10441667 0.09751667 0.09915   ]]
"""
'''
p = int(math.sqrt(Train_X.shape[1]))
shape =  (p, p)

"""
From the observation of frequencies of Train_y data, it's seen that the percentage of each class (10) is all around 10%;
it hence suggests a "balanced" data set.
As a result, for the evaluation of K-NN, I directly use "Accuracy" metric.
"""

''' 
#1.2 Data inspection and visualization
print('Dimensions: %s x %s' % (Train_X.shape[0], Train_X.shape[1]))
print('\n1st row', Train_X[0])
print('\n1st row', Train_y[0])


p = int(math.sqrt(Train_X.shape[1]))
shape =  (p, p)

## Visualization in 28*28 image form
i=1
img_c = Train_X[i,:].reshape(shape)
#img_c = Train_X[i,:].reshape(shape,order="F") #arg name "order" must be explicit or an error occurs!
# or np.reshape(d[:,i], shape, "F")
# Display the image
plt.imshow(img_c, cmap='gray')
# Hide the axes
plt.axis('off')
plt.title("Case: {}; Label: {}".format(i,Train_y[i]))
plt.show()
'''

#1.2 Select data (X, y)

Train_X_0 = Train_X[Train_y.values == 0] # N_0 = 5923; (5923*0.05)/(0.95*9) = 34.637426900584806
Train_y_0 = Train_y[Train_y.values == 0] 



df_X = pd.DataFrame([],columns = Train_X.columns)
df_y = pd.DataFrame([],columns = Train_y.columns)
for i in range(1,10): #range(1,10):
    df_X_ab = Train_X[Train_y.values == i][:34] 
    df_y_ab = Train_y[Train_y.values == i][:34] 
    
    df_X = df_X.append(df_X_ab)
    df_y = df_y.append(df_y_ab)

Train_X = Train_X_0.append(df_X) # N = 5923 + 9*34 = 6229
Train_y = Train_y_0.append(df_y)

#================================= 1. PCA for dimension reduction (Train) ========================================================================
# ---------------------------------------------------------------------

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Train_X = sc_X.fit_transform(Train_X)
Test_X = sc_X.transform(Test_X) #standardized X_test with the scales(mean/sd) derived form X_train


## Dimension Reduction: PCA

from sklearn.decomposition import PCA
'''
k = 100
pca = PCA(n_components = k, whiten=False)  #to explore all eigenvectors
'''

pca = PCA(n_components=.85)  #choose the minimum number of principal components such that 85% of the variance is retained.

Train_W = pca.fit_transform(Train_X)
Test_W = pca.transform(Test_X)

# Eigenvectors
U = pca.components_.T

feature_name = ["F"+str(i) for i in range(U.shape[1])]
U = pd.DataFrame(U, columns = feature_name )

## Projected data
Train_W = pd.DataFrame(Train_W, columns = feature_name )
Test_W = pd.DataFrame(Test_W, columns = feature_name )

Train_W.to_csv("Feature_extracted_via_PCA_dim({})_MNIST(W_Train).csv".format(U.shape[1]))
Test_W.to_csv("Feature_extracted_via_PCA_dim({})_MNIST(W_Test).csv".format(U.shape[1]))


## Visualization of cumulated explained_variance 
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize = (50,25))
plt.plot(explained_variance.cumsum())
plt.show()



## Visualization of W at the first 3 dim.
from plotly.offline import plot # to make plotly work in spyder, add this line and replace "fig.show()" with plot(fig)
import plotly.express as px

df = pd.concat([Train_W, Train_y],axis =1)
fig = px.scatter_3d(df, x='F0', y='F1', z='F2', color='digit')
fig.update_traces(marker_size = 5)
#fig.show()
plot(fig)
fig.write_html("./Output/"+"3D Scatter plot after PCA.html")



## Visualization in 28*28 image form - PC (features of "Normal" by subspace)
i_Max = 30 #PC_i - starting from 1: PC1~ 

for i in range(1,i_Max+1):
    img_c = U.iloc[:,i-1].values.reshape(shape)
    #img_c = Train_X[i,:].reshape(shape,order="F") #arg name "order" must be explicit or an error occurs!
    # or np.reshape(d[:,i], shape, "F")
    # Display the image
    plt.imshow(img_c, cmap='gray')
    # Hide the axes
    plt.axis('off')
    plt.title("Feature of {}-th PC".format(i))
    plt.show()
    # save the image
    #plt.imsave("./Output/"+f"Feature of {i}-th PC.png", img_c, cmap='gray') # or similarly below (with title attached)
    plt.savefig("./Output/"+f"Feature of {i}-th PC.png")
    plt.close()

