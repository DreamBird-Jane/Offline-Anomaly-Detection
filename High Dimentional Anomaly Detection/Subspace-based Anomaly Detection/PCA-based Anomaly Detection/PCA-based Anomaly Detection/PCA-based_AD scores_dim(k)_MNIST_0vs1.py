#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:39:39 2022

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






#============================================================= 1. Experiments ========================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import PCAbased_AD_Algorithms as PCA_AD
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
#import operator
import random
random.seed(a=0) #fix a to a specific number, so the experiment can be replicated <----------------


def FPrate(y, y_pred):
    N1 = y[y == 1].sum()
    N0 = len(y) - N1
    cm = confusion_matrix(y, y_pred)
    return  cm[0, 1]/N0


## Seting
## Define normal diget: d_normal
d_normal = 0  #<----------------------------------------- you can define number as normal
N0 = 5700
Train_X_0 = Train_X[Train_y.values == d_normal]

## Define abnormal digit: d_ab
d_ab = 1      #<----------------------------------------- you can define number as anomalies other than d_normal
N_ab = 300
Train_X_ab = Train_X[Train_y.values == d_ab]

N = N0 + N_ab

# For experiment and anomaly detection classification
n_iter = 5 # #{iterations}
#ks = [1, 784]
ks = list(range(1,17))+[ 2** p for p in range(5,10)]+[784] # #{PCs}
threshold = 0.05

##Set the output folder name (fold should be put subordinate to cwdir)
out_folder = f'Out_{d_normal}vs{d_ab}'
out_dir = os.path.join(cwdir,out_folder)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
#------------------------------------------------------------------------------------------------------------------------

outF_Lk = open(os.path.join(out_dir, f"Evaluation_Lk_{d_normal}vs{d_ab}.csv"), "a", newline='\n')
outF_Tk = open(os.path.join(out_dir, f"Evaluation_Tk_{d_normal}vs{d_ab}.csv"), "a", newline='\n')


for it in range(n_iter):
    #1. Random select data (X, y)+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Random select normal data
    indices_select = random.sample( list(range(0,Train_X_0.shape[0])),  N0 ) 
    Train_X_0_sel = Train_X_0.iloc[indices_select].copy()  
    Train_y_0 = pd.DataFrame(np.zeros(N0), columns=['Is_Ab'])#Train_y[Train_y.values == d_normal][:N0] 
    Train_y_0.index = Train_X_0_sel.index
    
    ## Random select abnormal data
    indices_select = random.sample( list(range(0,Train_X_ab.shape[0])),  N_ab ) 
    Train_X_ab_sel = Train_X_ab.iloc[indices_select]
    Train_y_ab = pd.DataFrame(np.ones(N_ab), columns=['Is_Ab'])#Train_y[Train_y.values == d_ab][:N_ab ] 
    Train_y_ab.index = Train_X_ab_sel.index
    
    ## Selected data for analysis
    X = Train_X_0_sel.append(Train_X_ab_sel) # N = 5700 + 300 = 6000
    y = Train_y_0.append(Train_y_ab)
    unique_elements, counts_elements = np.unique(y, return_counts=True)
    print("Frequency of unique values of the said array:")
    print( np.asarray((unique_elements, counts_elements)) )
    del Train_X_0_sel, Train_X_ab_sel, Train_y_0, Train_y_ab, indices_select
    
    
    for k in ks:
        print(f'iteration {it},\t k={k}')
        start2 = time.time() #<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        #2. Dimension Reduction: PCA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #2.1 Feature Scaling
        sc_X = StandardScaler()
        X_std = sc_X.fit_transform(X)
        
        #2.2. PCA
        #pca = PCA(n_components= k)  #for calculate rank-k leverage scores #<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pca = PCA()  #for calculate rank-k projection scores #<<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        W = pca.fit_transform(X_std) # dim = (n, k); k <= p
        V = pca.components_.T #dim = (p, k); k <= p
        s2 = pca.explained_variance_ #len = k <= p
        
        
        feature_name = ["PC"+str(i+1) for i in range(V.shape[1])]
        W = pd.DataFrame(W, columns = feature_name, index = X.index ) # W = np.dot(X_std, V)
        V = pd.DataFrame(V, columns = feature_name )
        
        
        #3. Anomaly Detection: Calculate outlierness +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++      
        """
        #3.1. For Rank-k Leverage Scores --------------------------------------------------------------------------------
        ##3.1.1. Calculate outlier / anomaly scores 
        AD_scores = W.apply(PCA_AD.Rank_k_leverage_score_y,axis=1,args=[k,s2])
        #AD_scores = pd.DataFrame(AD_scores) #, index = X.index
        #AD_scores.index = X.index
        
        ##3.1.2. Dichotomize scores into {normal(0), abnormal(1)} with simple threshold
        AD_binary = PCA_AD.predict_binary(AD_scores, threshold = threshold)
        AD_binary = pd.Series(AD_binary, index = AD_scores.index)
        
        #3.1.3 Ad-hoc Evaluations 
        ### Efficacy
        cm = confusion_matrix(y, AD_binary);print("confusion_matrix:\n{}".format(cm))
        FP = FPrate(y, AD_binary).values[0]#;print("FP rate:\n{}".format(FP))        
        acc = accuracy_score(y, AD_binary)#;print("accuarcy: {}".format(acc))
        f1 = f1_score(y, AD_binary)#;print("f1 score: {}".format(f1))
        precision = precision_score(y, AD_binary)#;print("precision score: {}".format(precision))
        recall = recall_score(y, AD_binary)#;print("recall score: {}".format(recall))        
        
        ### Efficiency
        end2 = time.time()
        t_it = end2 - start2#;print(end2 - start2,'sec. \n') #unit: sec.
        
        ### Output
        record = pd.DataFrame([{"digit_Ab": d_ab, "iteration": it, "k": k, 
                                "FPrate": FP, "accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "time": t_it,
                                }])                
        print(outF_Lk.tell())
        record.to_csv(outF_Lk,mode='a',header = outF_Lk.tell()==0, index=False)
        
        """
        #3.2. For Rank-k Projection Scores --------------------------------------------------------------------------------
        ##3.2.1. Calculate outlier / anomaly scores         
        AD_scores = W.apply(PCA_AD.Rank_k_projection_distance_y,axis=1,args=[k]) 
        #AD_scores = pd.DataFrame(AD_scores) #, index = X.index
        #AD_scores.index = X.index
        
        ##3.2.2. Dichotomize scores into {normal(0), abnormal(1)} with simple threshold
        AD_binary = PCA_AD.predict_binary(AD_scores, threshold = threshold)
        AD_binary = pd.Series(AD_binary, index = AD_scores.index)
        
        #3.2.3 Ad-hoc Evaluations 
        ### Efficacy
        cm = confusion_matrix(y, AD_binary);print("confusion_matrix:\n{}".format(cm))
        FP = FPrate(y, AD_binary).values[0]#;print("FP rate:\n{}".format(FP))        
        acc = accuracy_score(y, AD_binary)#;print("accuarcy: {}".format(acc))
        f1 = f1_score(y, AD_binary)#;print("f1 score: {}".format(f1))
        precision = precision_score(y, AD_binary)#;print("precision score: {}".format(precision))
        recall = recall_score(y, AD_binary)#;print("recall score: {}".format(recall))        
        
        ### Efficiency
        end3 = time.time()
        t_it = end3 - start2#;print(end2 - start2,'sec. \n') #unit: sec.
        
        ### Output
        record = pd.DataFrame([{"digit_Ab": d_ab, "iteration": it, "k": k,
                                "FPrate": FP, "accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "time": t_it,
                                }])                
        print(outF_Tk.tell())
        record.to_csv(outF_Tk,mode='a',header = outF_Tk.tell()==0, index=False)
        
        
        del X_std, W, V, s2, AD_scores, AD_binary, record
        
    del X, y

    
outF_Lk.close()
outF_Tk.close()






#============================================================= 2. Result  Visualization ========================================================================
Record_Lk = pd.read_csv(os.path.join(out_dir, f"Evaluation_Lk_{d_normal}vs{d_ab}.csv"))
Record_Tk = pd.read_csv(os.path.join(out_dir, f"Evaluation_Tk_{d_normal}vs{d_ab}.csv"))
Record_Lk['metric'] = 'Lk'
Record_Tk['metric'] = 'Tk'
Record = pd.concat([Record_Lk, Record_Tk], axis = 0)


Record_agg = Record.groupby(['digit_Ab','metric', 'k']).agg([np.mean, np.std])


## Acc & F1 ---------------------------------------------------------------------------
fig, ax = plt.subplots()
#ax2 = ax.twinx()

x_axis = ks[:16] #Record_agg.MultiIndex.get_level_values('k')

# Accuracy scoress
y_axis = Record_agg.loc[(d_ab, 'Lk'),('accuracy', 'mean')][:16]
yerr = Record_agg.loc[(d_ab, 'Lk'),('accuracy', 'std')][:16]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='g', lw=1, label='Acc-Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('accuracy', 'mean')][:16]
yerr = Record_agg.loc[(d_ab, 'Tk'),('accuracy', 'std')][:16]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-.', lw=1, c='g', label='Acc-Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('accuracy', 'mean')]
ax.axhline(mah, color='g', linestyle=':', lw=1, label='Acc-L784')

# F1 scoress
y_axis = Record_agg.loc[(d_ab, 'Lk'),('f1', 'mean')][:16]
yerr = Record_agg.loc[(d_ab, 'Lk'),('f1', 'std')][:16]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='blue', lw=1, label='F1-Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('f1', 'mean')][:16]
yerr = Record_agg.loc[(d_ab, 'Tk'),('f1', 'std')][:16]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-.', lw=1, c='blue', label='F1-Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('f1', 'mean')]
ax.axhline(mah, color='blue', linestyle=':', lw=1, label='F1-L784')

ax.set_xticks(x_axis)
ax.set_yticks(np.arange(0,1.01,0.05))
ax.set_xlabel("k (#{PCs})")
ax.set_ylabel("Metrics")

ax.legend() #loc='middle right'
plt.savefig(os.path.join(out_dir, "Evaluation restuls_Acc_F1.png"), transparent=True)
plt.show()


## Recall & Recall ---------------------------------------------------------------------------
fig, ax = plt.subplots(1,2, figsize=(10,5), sharey=True)#
#ax2 = ax.twinx()

x_axis = ks[:16] #Record_agg.MultiIndex.get_level_values('k')

# Recall
y_axis = Record_agg.loc[(d_ab, 'Lk'),('recall', 'mean')][:16]
yerr = Record_agg.loc[(d_ab, 'Lk'),('recall', 'std')][:16]
ax[0].errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='b', lw=1, label='Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('recall', 'mean')][:16]
yerr = Record_agg.loc[(d_ab, 'Tk'),('recall', 'std')][:16]
ax[0].errorbar(x_axis, y_axis , yerr=yerr, ls='-.', c='r', lw=1, label='Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('recall', 'mean')]
ax[0].axhline(mah, color='b', linestyle=':', lw=1, label='L784')

ax[0].set_yticks(np.arange(0,1.01,0.05))
ax[0].set_xticks(x_axis)
ax[0].set_xlabel("k (#{PCs})")
ax[0].set_ylabel("Recall")
ax[0].legend()

# Precsion
y_axis = Record_agg.loc[(d_ab, 'Lk'),('precision', 'mean')][:16]
yerr = Record_agg.loc[(d_ab, 'Lk'),('precision', 'std')][:16]
ax[1].errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='b', lw=1, label='Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('precision', 'mean')][:16]
yerr = Record_agg.loc[(d_ab, 'Tk'),('precision', 'std')][:16]
ax[1].errorbar(x_axis, y_axis , yerr=yerr, ls='-.', c='r', lw=1, label='Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('precision', 'mean')]
ax[1].axhline(mah, color='b', linestyle=':', lw=1, label='L784')

ax[1].set_xticks(x_axis)
ax[1].set_xlabel("k (#{PCs})")
ax[1].set_ylabel("Precision")
ax[1].legend() #loc='middle right'
plt.savefig(os.path.join(out_dir, "Evaluation restuls_Recall_Precision.png"), transparent=True)

plt.show()



# 3.1 Full PCS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Acc & F1 ---------------------------------------------------------------------------
fig, ax = plt.subplots()
#ax2 = ax.twinx()

x_axis = ks #Record_agg.MultiIndex.get_level_values('k')

# Accuracy scoress
y_axis = Record_agg.loc[(d_ab, 'Lk'),('accuracy', 'mean')]
yerr = Record_agg.loc[(d_ab, 'Lk'),('accuracy', 'std')]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='g', lw=1, label='Acc-Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('accuracy', 'mean')]
yerr = Record_agg.loc[(d_ab, 'Tk'),('accuracy', 'std')]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-.', lw=1, c='g', label='Acc-Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('accuracy', 'mean')]
ax.axhline(mah, color='g', linestyle=':', lw=1, label='Acc-L784')

# F1 scoress
y_axis = Record_agg.loc[(d_ab, 'Lk'),('f1', 'mean')]
yerr = Record_agg.loc[(d_ab, 'Lk'),('f1', 'std')]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='blue', lw=1, label='F1-Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('f1', 'mean')]
yerr = Record_agg.loc[(d_ab, 'Tk'),('f1', 'std')]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-.', lw=1, c='blue', label='F1-Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('f1', 'mean')]
ax.axhline(mah, color='blue', linestyle=':', lw=1, label='F1-L784')

ax.set_xticks(x_axis)
ax.set_yticks(np.arange(0,1.01,0.05))
ax.set_xlabel("k (#{PCs})")
ax.set_ylabel("Metrics")

ax.legend() #loc='middle right'
plt.savefig(os.path.join(out_dir, "Evaluation restuls_Acc_F1_all PCs.png"), transparent=True)
plt.show()


## Recall & Recall ---------------------------------------------------------------------------
fig, ax = plt.subplots(2,1, figsize=(10,10), sharey=True, sharex=True)#
#ax2 = ax.twinx()

x_axis = ks #Record_agg.MultiIndex.get_level_values('k')

# Recall
y_axis = Record_agg.loc[(d_ab, 'Lk'),('recall', 'mean')]
yerr = Record_agg.loc[(d_ab, 'Lk'),('recall', 'std')]
ax[0].errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='b', lw=1, label='Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('recall', 'mean')]
yerr = Record_agg.loc[(d_ab, 'Tk'),('recall', 'std')]
ax[0].errorbar(x_axis, y_axis , yerr=yerr, ls='-.', c='r', lw=1, label='Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('recall', 'mean')]
ax[0].axhline(mah, color='b', linestyle=':', lw=1, label='L784')

ax[0].set_yticks(np.arange(0,1.01,0.05))
ax[0].set_xticks(x_axis)
ax[0].set_xlabel("k (#{PCs})")
ax[0].set_ylabel("Recall")
ax[0].legend()

# Precsion
y_axis = Record_agg.loc[(d_ab, 'Lk'),('precision', 'mean')]
yerr = Record_agg.loc[(d_ab, 'Lk'),('precision', 'std')]
ax[1].errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='b', lw=1, label='Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('precision', 'mean')]
yerr = Record_agg.loc[(d_ab, 'Tk'),('precision', 'std')]
ax[1].errorbar(x_axis, y_axis , yerr=yerr, ls='-.', c='r', lw=1, label='Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('precision', 'mean')]
ax[1].axhline(mah, color='b', linestyle=':', lw=1, label='L784')

ax[1].set_xticks(x_axis)
ax[1].set_xlabel("k (#{PCs})")
ax[1].set_ylabel("Precision")
ax[1].legend() #loc='middle right'
plt.savefig(os.path.join(out_dir, "Evaluation restuls_Recall_Precision_all PCs.png"), transparent=True)

plt.show()




















'''
## Acc & F1 ---------------------------------------------------------------------------
fig, ax = plt.subplots()
#ax2 = ax.twinx()

x_axis = ks[:-2] #Record_agg.MultiIndex.get_level_values('k')

# Accuracy scoress
y_axis = Record_agg.loc[(d_ab, 'Lk'),('accuracy', 'mean')][:-2]
yerr = Record_agg.loc[(d_ab, 'Lk'),('accuracy', 'std')][:-2]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='g', lw=1, label='Acc-Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('accuracy', 'mean')][:-2]
yerr = Record_agg.loc[(d_ab, 'Tk'),('accuracy', 'std')][:-2]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-.', lw=1, c='g', label='Acc-Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('accuracy', 'mean')]
ax.axhline(mah, color='g', linestyle=':', lw=1, label='Acc-L784')

# F1 scoress
y_axis = Record_agg.loc[(d_ab, 'Lk'),('f1', 'mean')][:-2]
yerr = Record_agg.loc[(d_ab, 'Lk'),('f1', 'std')][:-2]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='blue', lw=1, label='F1-Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('f1', 'mean')][:-2]
yerr = Record_agg.loc[(d_ab, 'Tk'),('f1', 'std')][:-2]
ax.errorbar(x_axis, y_axis , yerr=yerr, ls='-.', lw=1, c='blue', label='F1-Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('f1', 'mean')]
ax.axhline(mah, color='blue', linestyle=':', lw=1, label='F1-L784')

ax.set_xticks(x_axis)
ax.set_yticks(np.arange(0,1.01,0.05))
ax.set_xlabel("k (#{PCs})")
ax.set_ylabel("Metrics")

ax.legend() #loc='middle right'
plt.savefig(os.path.join(out_dir, "Evaluation restuls_Acc_F1.png"), transparent=True)
plt.show()


## Recall & Recall ---------------------------------------------------------------------------
fig, ax = plt.subplots(2,1, figsize=(10,5), sharey=True, sharex=True)#
#ax2 = ax.twinx()

x_axis = ks[:-2] #Record_agg.MultiIndex.get_level_values('k')

# Recall
y_axis = Record_agg.loc[(d_ab, 'Lk'),('recall', 'mean')][:-2]
yerr = Record_agg.loc[(d_ab, 'Lk'),('recall', 'std')][:-2]
ax[0].errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='b', lw=1, label='Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('recall', 'mean')][:-2]
yerr = Record_agg.loc[(d_ab, 'Tk'),('recall', 'std')][:-2]
ax[0].errorbar(x_axis, y_axis , yerr=yerr, ls='-.', c='r', lw=1, label='Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('recall', 'mean')]
ax[0].axhline(mah, color='b', linestyle=':', lw=1, label='L784')

ax[0].set_yticks(np.arange(0,1.01,0.05))
ax[0].set_xticks(x_axis)
ax[0].set_xlabel("k (#{PCs})")
ax[0].set_ylabel("Recall")
ax[0].legend()

# Precsion
y_axis = Record_agg.loc[(d_ab, 'Lk'),('precision', 'mean')][:-2]
yerr = Record_agg.loc[(d_ab, 'Lk'),('precision', 'std')][:-2]
ax[1].errorbar(x_axis, y_axis , yerr=yerr, ls='-', c='b', lw=1, label='Lk')

y_axis = Record_agg.loc[(d_ab, 'Tk'),('precision', 'mean')][:-2]
yerr = Record_agg.loc[(d_ab, 'Tk'),('precision', 'std')][:-2]
ax[1].errorbar(x_axis, y_axis , yerr=yerr, ls='-.', c='r', lw=1, label='Tk')

mah = Record_agg.loc[(d_ab, 'Tk', 784),('precision', 'mean')]
ax[1].axhline(mah, color='b', linestyle=':', lw=1, label='L784')

ax[1].set_xticks(x_axis)
ax[1].set_xlabel("k (#{PCs})")
ax[1].set_ylabel("Precision")
ax[1].legend() #loc='middle right'
plt.savefig(os.path.join(out_dir, "Evaluation restuls_Recall_Precision.png"), transparent=True)

plt.show()
'''
