# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:21:06 2021

@author: Shixiong Jiang
bias vs variance
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report
#%%
def load_data(path):
    d = sio.loadmat(path)
    return map(np.ravel, [d['X'], d['y'], d['Xtest'], d['ytest'], d['Xval'], d['yval']]) 

X, y, Xtest, ytest, Xval, yval = load_data('ex5data1.mat')
df = pd.DataFrame({'water_level':X, 'flow':y})
sns.lmplot('water_level', 'flow', data = df, fit_reg=False)
plt.show()
X, Xtest, Xval = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis = 1)for x in (X, Xtest, Xval)]
#%%
def prepare_data(path, power):
    X, y, Xtest, ytest, Xval, yval = load_data(path)
    X, Xtest, Xval = [poly(x, power) for x in (X, Xtest, Xval)]
    X, Xtest, Xval = [normalize_feature(pd.DataFrame(x)) for x in (X, Xtest, Xval)]
    X, Xtest, Xval = [np.array(x) for x in (X, Xtest, Xval)]
    X, Xtest, Xval =[np.insert(x, 0, np.ones(x.shape[0]), axis = 1)for x in (X, Xtest, Xval)]
    return X, Xtest, Xval, y, ytest, yval
    
def cost(theta, X, y):
    m = X.shape[0]

    inner = X @ theta - y  # R(m*1)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost

def cost_regularize(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()

    return cost(theta, X, y) + regularized_term

def gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()  # same shape as theta
    regularized_term[0] = 0  # don't regularize intercept theta
    regularized_term = (l / m) * regularized_term
    return gradient_term(theta, X, y) + regularized_term

def gradient_term(theta, X, y):
    m = X.shape[0]

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)

    return inner / m

def logistic_regression(X, y, lamda = 1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=cost_regularize, x0=theta,args=(X, y, lamda), method='TNC', jac=gradient, options={'disp': True})
    final_theta = res.x
    return final_theta

def bia_variance(X, y, Xval, yval, l=1):
    cost_train = []
    cost_val = []
    for i in range(X.shape[0]):
        final_theta = logistic_regression(X[0:i+1], y[0:i+1],l)
        cost_train.append(cost_regularize(final_theta, X[0:i+1], y[0:i+1], l))
        cost_val.append(cost_regularize(final_theta, Xval, yval, l))
    return cost_train, cost_val

def poly(X, p):
    k = X
    init = k
    for i in range(2, p + 1):
        k = k * init
        X = np.insert(X.reshape(X.shape[0], i - 1), i-1, values=k, axis=1)
    return X
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())
def plot_learning_curve(X, y, Xval, yval, lamda=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = logistic_regression(X[:i, :], y[:i], lamda=lamda)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost_regularize(res, X[:i, :], y[:i], lamda)
        cv = cost_regularize(res, Xval, yval, lamda)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)
#%%draw bias and variance
theta = np.ones(X.shape[1])
final_theta1 = logistic_regression( X, y, 0)
ans = gradient(theta, X, y, 0)
m = np.arange(-50, 40, 1)
sns.lmplot('water_level', 'flow', data = df, fit_reg=False)
plt.plot(m, m * final_theta1[1] + final_theta1[0] )
plt.show()
#%%

#%%
size = np.arange(0, 12, 1)
cost_train, cost_val = bia_variance(X, y, Xval, yval, 0)
plt.plot(cost_train)
plt.plot(cost_val)
plt.show()
#%%
#X, Xtest, Xval = prepare_data('ex5data1.mat', 8)
#X, y, Xtest, ytest, Xval, yval = load_data('ex5data1.mat')
X, y, Xtest, ytest, Xval, yval = load_data('ex5data1.mat')
#%%
X, Xtest, Xval, y, ytest, yval = prepare_data('ex5data1.mat', 8)
#%%
plot_learning_curve(X, y, Xval, yval, 0)
#%%
plot_learning_curve(X, y, Xval, yval, 1)
#%%
plot_learning_curve(X, y, Xval, yval, 100)