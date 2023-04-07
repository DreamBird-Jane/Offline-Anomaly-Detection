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
import os

#Set the current working directory / or manually type-in
cwdir = os.getcwd() #'/Users/jane_hsieh/Library/CloudStorage/OneDrive-國立陽明交通大學/Data Science Analysis Templates/Anomaly Detection/Subspace-based Anomaly Detection/PCA-based Anomaly Detection'
#or manually type-in
cwdir = "/Users/jane_hsieh/Library/CloudStorage/OneDrive-國立陽明交通大學/Data Science Analysis Templates/Anomaly Detection/Subspace-based Anomaly Detection/PCA-based Anomaly Detection"
os.chdir(cwdir)

#indicate the directory path for the fold where 4 MNIST data are located (default: in cwdir)
data_dir = './MNIST Database'

#Set the output folder name (fold should be put subordinate to cwdir)
out_folder = 'Out_0vs1'
out_dir = os.path.join(cwdir,out_folder)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)



"""
Steps:
    0. Input MNIST dataset
    1. Since the dimension of the feature data is too high, I hence do spectral clustering for dimension reduction, to save times
        1.1 Calculate affinity matrix A: s(x, y)= rbf_kernel similarity if class(x)=class(y); o.w. s(x, y)= 0; and double scale A => L
        1.2. Feature extraction via spectral dimension reduction: X (dim = Train_N x p) = > W (dim = Train_N x k)
    2. Do the same spectral dimension reduction for Test data
"""
# ================================ Functions ===================================================
def plt_img(X, y, i=0, shape=(28,28), save=True, sc=False):
    '''
    Image Visualization in 28*28 image form
    Parameters
    ----------
    X : image data; DESCRIPTION.
    y : label/digit data; DESCRIPTION.
    i : sample index, optional; DESCRIPTION. The default is 0.
    shape : sie of image, optional;DESCRIPTION. The default is (28,28).

    Returns
    -------
    None.

    '''
    img_c = X[i,:].reshape(shape)
    plt.imshow(img_c, cmap='gray')
    plt.axis('off') # Hide the axes
    plt.title("Case: {}; Label: {}".format(i,int(y[i])))
    
    if save:
        if sc:
            plt.savefig( os.path.join(out_dir, f"Sample {i} with digit {int(y[i])}(sc).png"))
        else:
            plt.savefig( os.path.join(out_dir, f"Sample {i} with digit {int(y[i])}.png"))
            
    plt.show()
    return



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
Test_X = pd.DataFrame(Test_X)
Test_y = pd.DataFrame(Test_y, columns = ["digit"])


## Inspenct the frequency of each class (digit 0-9)
unique_elements, counts_elements = np.unique(Train_y, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


Train_N = Train_X.shape[0]
print("Percentage of each class of Train_y data: \n {}".format(np.asarray((unique_elements, counts_elements/Train_N))))

"""
    Frequency of unique values of the said array:
    [[   0    1    2    3    4    5    6    7    8    9]
     [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]]
    Percentage of each class of Train_y data: 
     [[0.         1.         2.         3.         4.         5.         6.         7.         8.         9.        ]
     [0.09871667 0.11236667 0.0993     0.10218333 0.09736667 0.09035     0.09863333 0.10441667 0.09751667 0.09915   ]]
     
    From the observation of frequencies of Train_y data, it's seen that the percentage of each class (10) is all around 10%;
     it hence suggests a "balanced" data set.
     As a result, for the evaluation of K-NN, I directly use "Accuracy" metric.

"""
p = Train_X.shape[1]
width = int(math.sqrt(p))
shape =  (width, width)



''' 
#1.2 Data inspection and visualization
print('Dimensions: %s x %s' % (Train_X.shape[0], Train_X.shape[1]))
print('\n1st row', Train_X.iloc[0,:])
print('\n1st row', Train_y.iloc[0])

'''

#1.3 Select data (X, y)
import random
random.seed(a=0) #fix a to a specific number, so the experiment can be replicated

## Define normal diget: d_normal 
d_normal = 0  #<------------------------------------------------ you can define number as normal
N0 = 5700
Train_X_0 = Train_X[Train_y.values == d_normal]
indices_select = random.sample( list(range(0,Train_X_0.shape[0])),  N0 ) 
Train_X_0 = Train_X_0.iloc[indices_select]

Train_y_0 = Train_y[Train_y.values == d_normal][:N0] 
Train_y_0.index = Train_X_0.index


## Define abnormal digit: d_ab
d_ab = 1      #<----------------------------------------- you can define number as anomalies other than d_normal
N_ab = 300
Train_X_ab = Train_X[Train_y.values == d_ab]
indices_select = random.sample( list(range(0,Train_X_ab.shape[0])),  N_ab ) 
Train_X_ab = Train_X_ab.iloc[indices_select]

Train_y_ab = Train_y[Train_y.values == d_ab][:N_ab ] 
Train_y_ab.index = Train_X_ab.index

## Selected data for analysis
X = Train_X_0.append(Train_X_ab) # N = 5700 + 300 = 6000
y = Train_y_0.append(Train_y_ab)
unique_elements, counts_elements = np.unique(y, return_counts=True)
print("Frequency of unique values of the said array:")
print( np.asarray((unique_elements, counts_elements)) )

# inspect data instance i
plt_img(X.values, y.values, i=0)


del Train_X_0, Train_X_ab, Train_y_0, Train_y_ab

'''
## Abnormal data: digit 1~9

df_X = pd.DataFrame([],columns = Train_X.columns)
df_y = pd.DataFrame([],columns = Train_y.columns)
N_abs = [34, 34, 34, 33, 33, 33, 33, 33, 33]
for i in range(1,10): 
    df_X_ab = Train_X[Train_y.values == i]
    indices_select = random.sample( list(range(0,df_X_ab.shape[0])),  N_abs[i-1] ) 

    df_X_ab = df_X_ab.iloc[indices_select]
    df_y_ab = Train_y[Train_y.values == i][:N_abs[i-1]] 
    
    df_X = df_X.append(df_X_ab)
    df_y = df_y.append(df_y_ab)

X = Train_X_0.append(df_X) # N = 5700 + 34*3 + 33*6 = 6000
y = Train_y_0.append(df_y)

del df_X, df_y, df_X_ab, df_y_ab
'''



#================================= 1. PCA for dimension reduction (Train) ========================================================================
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------
#1.0 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_std = sc_X.fit_transform(X)
#Test_X = sc_X.transform(Test_X) #standardized X_test with the scales(mean/sd) derived form X_train
plt_img(X_std, y.values, i=5999, sc=True)

## Dimension Reduction: PCA
#1.1 Explore number of PCs - k using elbow methods


# Define the range of PCs to try
PCs = range(1, p+1)
pca = PCA()  #default: n_components= p**2 (all dim.)
#pca = PCA(n_components=.85)  #choose the minimum number of principal components such that 85% of the variance is retained.



EVRs = pca.fit(X_std).explained_variance_ratio_
EVRs = pca.explained_variance_ratio_
EVRs_cumsum = np.cumsum(EVRs)

EVRs_cumsum = pd.DataFrame(EVRs_cumsum) 
'''
Cumulative EVR = 80% when k=74 (index=73)
Cumulative EVR = 85% when k=94 (index=93) <<<<< -------  decides number of k = 94
Cumulative EVR = 90% when k=125 (index=124)
'''
EVRs_cumsum.to_csv(os.path.join(out_dir, "Explained Variance Vatio (cumsum).csv"))


# Plot the models and their respective score 
colors = ['red', 'orange', 'green']
plt.figure(figsize=(15, 8))
#plt.plot(PCs[:200], EVRs[:200])
plt.plot(PCs, EVRs_cumsum, ) #[:200]
plt.xlabel('Number of PC')
plt.ylabel('Cumulative Explained Variance Ratio (EVR)')
plt.title('Elbow Curve')
plt.xticks(np.arange(0,783,25))
plt.yticks(np.arange(0,1,0.05))
# Add a vertical/horizontal line
for i, j in enumerate([74, 94,125]):
    plt.axvline(j, color=colors[i], linestyle='--')
    plt.annotate(str(j), xy = (j, 0), xytext = (j-7, 0.01))
for i, j in enumerate([0.8, 0.85, 0.9]):
    plt.axhline(j, color=colors[i], linestyle='--') # 4 in range(y)

plt.savefig( os.path.join(out_dir, "Elbow Curve.png"))
plt.show()






#================================= 2. Explore "Normal pattern" from PCA perspectives========================================================================
# Eigenvectors
V = pca.components_.T #dim = (p, k); k <= p

feature_name = ["PC"+str(i+1) for i in range(V.shape[1])]
V = pd.DataFrame(V, columns = feature_name )

# Eigenvalues s2 (variance of PCs)
s2 = pca.explained_variance_ #len = k <= p

#-------------------------------------- 2.1 Explore normal features of PCs --------------------------------------

## Visualization in 28*28 image form - PC (features of "Normal" by subspace)
"""
i_Max = 30 #PC_i - starting from 1: PC1~ 

for i in range(i_Max):
    img_c = V.iloc[:,i].values.reshape(shape)
    #img_c = X[i,:].reshape(shape,order="F") #arg name "order" must be explicit or an error occurs!
    # or np.reshape(d[:,i], shape, "F")
    # Display the image
    plt.imshow(img_c, cmap='gray')
    # Hide the axes
    plt.axis('off')
    plt.title("Feature of {}-th PC".format(i+1))
    plt.show()
    # save the image
    #plt.imsave("./Output/"+f"Feature of {i}-th PC.png", img_c, cmap='gray') # or similarly below (with title attached)
    plt.savefig( os.path.join(out_dir, f"Feature of {i+1}-th PC.png"))
    #plt.savefig("./Output/"+f"Feature of {i}-th PC.png")
    plt.close()
"""


rows = 5
columns = 10 

# first (1+shift) ~ (rows*columns+shift) PCs
shift = 0 #must be the multiplier of (rows * columns), e.g, 0, 50, 100,... #<------------------------------------------------
fig = plt.figure(figsize=(50, 100))
for i in range(1+shift, columns*rows +1+shift):
    img_c = V.iloc[:,i-1].values.reshape(shape)
    fig.add_subplot(rows, columns, i-shift)
    plt.imshow(img_c, cmap='gray')
    plt.axis('off')
    plt.title("PC{}".format(i), fontsize=6)
plt.show()
plt.savefig( os.path.join(out_dir, f"Features of {1+shift}~{rows*columns+shift} PCs.png"))


shift = 750 
fig = plt.figure(figsize=(50, 100))
for i in range(1+shift, p+1):
    img_c = V.iloc[:,i-1].values.reshape(shape)
    fig.add_subplot(rows, columns, i-shift)
    plt.imshow(img_c, cmap='gray')
    plt.axis('off')
    plt.title("PC{}".format(i), fontsize=6)
plt.show()
plt.savefig( os.path.join(out_dir, f"Features of {1+shift}~{p} PCs.png"))




#-------------------------------------- 2.2 Explore distrition PCs --------------------------------------
## Projected data
W = pca.transform(X_std)

W_y = np.concatenate([W, y], axis = 1)
W_y = pd.DataFrame(W_y, columns = feature_name+y.columns.tolist() )



import seaborn as sns

for shift in range(0, 50+1, 10):
#for shift in range(750, 770+1, 10):
    #shift = 0
    sns.pairplot(vars = ["PC"+str(i) for i in range(shift+1, 10+shift+1)], 
                 hue='digit', markers='.', plot_kws=dict(alpha=0.5),
                 corner=True, data = W_y) #, diag_kind = 'hist'
    plt.savefig( os.path.join(out_dir, f"Pairwise plots of {1+shift}~{10+shift} PCs (hue by digit).png"), transperent=True)

## for latest PCS
shift = 780#775
sns.pairplot(vars = ["PC"+str(i) for i in range(shift+1, p+1)], 
                     hue='digit', markers='.', plot_kws=dict(alpha=0.5),
                     corner=True, data = W_y) #, diag_kind = 'hist'
plt.savefig( os.path.join(out_dir, f"Pairwise plots of {1+shift}~{p} PCs (hue by digit).png"), transperent=True)
    





#-------------------------------------- 2.2 Explore projection of PCA in lower subspaces (3D plot) --------------------------------------
## 3D scatterplot in different PCS
from plotly.offline import plot # to make plotly work in spyder, add this line and replace "fig.show()" with plot(fig)
import plotly.express as px
import plotly.graph_objs as go

W_y2 = W_y.astype({'digit':str}) #in order to make the colorbar of scatter_3d as categorical rather than continuous
print("Check:\n", W_y2.dtypes)


from itertools import combinations

#stuff = ["PC"+str(i) for i in range(1,4+1)] #must be iterable
#stuff = ['PC1']+["PC"+str(i) for i in range(5,7+1) ] #must be iterable
#stuff = ['PC1','PC2']+["PC"+str(i) for i in range(782,784+1) ] #must be iterable
stuff = ["PC"+str(i) for i in range(781,784+1)] #must be iterable
for axis_x, axis_y, axis_z in combinations(stuff, 3): # L=3
    print(axis_x, axis_y, axis_z)
    #axis_x, axis_y, axis_z = subset
    fig = px.scatter_3d(W_y2, x=axis_x, y=axis_y, z=axis_z, color='digit')
    fig.update_traces(marker_size = 2)
    #fig.show()
    plot(fig)
    fig.write_html( os.path.join(out_dir, f"3D Scatter plot after PCA_({axis_x}, {axis_y}, {axis_z}).html") )
    
del W_y2



## 3D scatterplot in different PCS vs Normal Subspace (e.g., Span{PC1,PC2})
grids = np.linspace(-20, 20, 100)#np.arange(-20, 20.005, 0.05)
xx, yy = np.meshgrid(grids, grids)
zz = 0*xx

axis_x = 'PC1'
axis_y = 'PC2'
#axis_z = 'PC3'
for axis_z in ["PC"+str(i) for i in range(3,10+1)]+["PC"+str(i) for i in range(775,784+1)]:
    
    layout = go.Layout(title_text="PCA Projection vs Normal Subspace (e.g., Span{PC1,PC2})",
                       scene=dict(xaxis_title=axis_x, yaxis_title= axis_y, zaxis_title=axis_z)) #width = 1000, height =1000, 
    fig = go.Figure(data=[go.Surface(x = xx, y = yy, z=zz, colorscale = 'Blues', opacity=.5)], layout=layout)
    fig.update_traces(showscale=False)
    fig.add_scatter3d(x=W_y[axis_x], y=W_y[axis_y], z = W_y[axis_z], mode='markers',
                      marker=dict(size=2, color=W_y['digit'], colorscale=[(0,'blue'), (1, 'red')], opacity=0.5)) 
    plot(fig)
    fig.write_html( os.path.join(out_dir, f"3D Scatter plot after PCA_({axis_x}, {axis_y}, {axis_z})_vs Span({axis_x},{axis_y}).html") )







#================================= 3. Calculate outlier scores ========================================================================
import PCAbased_AD_Algorithms as PCA_AD
import time

#W = pca.transform(X_std)

feature_name = ["PC"+str(i+1) for i in range(X.shape[1])]
W = pd.DataFrame(W, columns = feature_name, index = X.index) # Y_pca = np.dot(X_std, V)



start2 = time.time() 

ks = [ 2** p for p in range(0,10)]+[784]

AD_scores = {}
for k in ks:
    print(k)

    name = 'L_'+str(k);print(name)
    AD_scores[name] = W.apply(PCA_AD.Rank_k_leverage_score_y,axis=1,args=[k,s2])
    name = 'T_'+str(k);print(name)
    AD_scores[name] = W.apply(PCA_AD.Rank_k_projection_distance_y,axis=1,args=[k])
    
end2 = time.time()
print(end2 - start2,'sec. \n') #unit: sec.
print('I.e., the running time is {:.3f} min.'.format((end2 - start2)/60))    


AD_scores = pd.DataFrame(AD_scores, index = X.index) #, index = X.index
AD_scores['digit'] = y
AD_scores.to_csv(os.path.join(out_dir, "ADscores_0vs1.csv"))





