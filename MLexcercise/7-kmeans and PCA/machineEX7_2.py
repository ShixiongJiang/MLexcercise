# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:50:41 2021

@author: Shixiong Jiang
K Means on Image to compress image
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
def load_data(path):
    mat = sio.loadmat(path)
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    return data
def visualize_data(data):
    sns.lmplot('X1', 'X2', data=data, fit_reg=False)
    plt.show()
    
def load_image(path):
    data = io.imread(path) / 255.
    io.imshow(data)
    return data
#%%
data = load_image('bird_small.png')
print(data.shape)

#%%
def FindClosetCentroids(centroids, x):
    distances = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=centroids-x)
    return np.argmin(distances)
def AssignCluster(data, centroids):
    return np.apply_along_axis(lambda x:FindClosetCentroids(centroids, x), axis=1, arr=data)
def combine_data(data, C):
    data_with_C = data.copy()
    data_with_C['C'] = C
    return data_with_C
def computeCentroids(data, C):
    data_with_C = combine_data(data, C)
    return data_with_C.groupby('C', as_index=False).mean().sort_values(by='C').drop('C', axis=1)
                       
def random_init(data, k):
    return np.array(data.sample(k))

def cost(data, centroids, C):
    distances = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=np.array(data) - np.array(centroids)[C])
    return distances.sum() / data.shape[0]

    
def iteration_for_k_means(data, k,epoch=100):
    centroids = random_init(data, k)
    C = AssignCluster(data, centroids)
    process_cost = []
    for i in range(epoch):
        print('running epoch{}'.format(i))
        centroids = computeCentroids(data, C)
        C = AssignCluster(data, centroids)
        process_cost.append(cost(data, centroids, C))
        if len(process_cost) > 1:
            if np.abs(process_cost[-1] - process_cost[-2]) / process_cost[-1] < 0.005:
                break
    return centroids, C, process_cost[-1]
def k_means(data, k, init=10):
    temp = np.array([iteration_for_k_means(data, k) for _ in range(init)])
    last_index = np.argmin(temp[:, -1])
    return temp[last_index]
#%%
data = data.reshape(128*128, 3)
ans = k_means(pd.DataFrame(data), 16)
centroids, C, cost = [ans[i] for i in range(3)]
#%%
centroids = np.array(centroids)
compressed_image = centroids[C]
compressed_image = compressed_image.reshape(128,128,3)
io.imshow(compressed_image)