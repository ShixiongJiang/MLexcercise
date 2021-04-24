# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:16:01 2021

@author: Shixiong Jiang
SVM
"""
import pandas as pd
import seaborn as sns #plot
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report
import sklearn.svm
#%%
def load_data(path):
    mat = sio.loadmat(path)
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    data['y'] = mat.get('y')
    return data
def Guaasian_kernel(X1, X2, sigma):
    return np.exp(-np.power(X1 - X2, 2).sum() / (2 * sigma * sigma))
#%%
data = load_data('ex6data1.mat')
sns.lmplot('X1', 'X2', data=data, fit_reg=False, hue='y')
plt.show()
#%%
svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge')
svc1.fit(data[['X1', 'X2']], data['y'])
svc1.score(data[['X1', 'X2']], data['y'])
data['confidence when c = 1'] = svc1.decision_function(data[['X1', 'X2']])
ax = plt.subplot()
ax.scatter(data['X1'], data['X2'], s=50, c=data['confidence when c = 1'], cmap='RdBu')
ax.set_title('SVM when C = 1')
plt.show()
#%%
svc2 = sklearn.svm.LinearSVC(C=100, loss='hinge')
svc2.fit(data[['X1', 'X2']], data['y'])
svc2.score(data[['X1', 'X2']], data['y'])
data['confidence when c = 100'] = svc2.decision_function(data[['X1', 'X2']])
ax = plt.subplot()
ax.scatter(data['X1'], data['X2'], s=50, c=data['confidence when c = 100'], cmap='RdBu')
ax.set_title('SVM when C = 100')
plt.show()
#%%
data = load_data('ex6data2.mat')
sns.lmplot('X1', 'X2', data = data, fit_reg=False, hue='y')
plt.show()
#%%
svc = sklearn.svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
svc.fit(data[['X1', 'X2']], data['y'])
ans = svc.score(data[['X1', 'X2']], data['y'])
predict_prob = predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap='RdBu')
plt.show()

#%%
