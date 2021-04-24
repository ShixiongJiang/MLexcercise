# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:15:35 2021

@author: Shixiong Jiang
PCA on two dimentional data
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report
import sklearn.svm
from sklearn.cluster import KMeans
from skimage import io
#%%
def normalize_features(X):
    X_copy = X.copy()
    m, n = X.shape
    for i in range(n):
        X_copy[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    return X_copy
def covariance_matrix(X):
    return (X.T @ X) / X.shape[0]
def pca(X):
    X_norm = normalize_features(X)
    sigma = covariance_matrix(X_norm)
    U,S,V = np.linalg.svd(sigma)
    return U,S,V
def project_data(X, U, k):
    n = X.shape[1]
    if k > n:
        raise ValueError("k shouldn't greater than n")
    return X @ U[:, :k]
def recover_data(Z, U):
    m, n = Z.shape
    if n > U.shape[0]:
        raise ValueError("n should't greater than k")
    return Z @ U[:, :n].T
#%%
mat = sio.loadmat('ex7data1.mat')
X = (mat.get('X'))
plt.scatter(X[:, 0], X[:, 1])
plt.show()
X_norm = normalize_features(X)
plt.scatter(X_norm[:, 0], X_norm[:, 1])
plt.show()
#%%
U, S, V = pca(X)
data_1d = project_data(X_norm, U, 1)
plt.plot(data_1d, len(data_1d) * [1], 'X')
plt.show()
#%%
X_recover = recover_data(data_1d, U)
plt.scatter(X_recover[:, 0], X_recover[:, 1])
plt.show()