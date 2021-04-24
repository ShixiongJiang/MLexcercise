# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 09:04:23 2021

@author: Shixiong Jiang
PCA on Face image
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
def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                    sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


#%%
mat = sio.loadmat('ex7faces.mat')
X = np.array([x.reshape(32, 32).T.reshape(1024) for x in mat.get('X')])
plot_n_image(X, 64)
plt.show()
#%%
U, S, V = pca(X)
plot_n_image(U, 64)
#%%
Z = project_data(X, U, 100)
plot_n_image(Z, 64)
#%%
X_recover = recover_data(Z, U)
plot_n_image(X_recover, 64)




