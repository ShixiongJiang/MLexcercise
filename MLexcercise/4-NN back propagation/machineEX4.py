# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:24:07 2021

@author: dell
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report
#%%
def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector

    X = data.get('X')  # (5000,400)

    if transpose:
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])

    return X, y
    
def plot_100_image(X):
    """ sample 100 image and show them
    assume the image is square

    X : (5000, 400)
    """
    size = int(np.sqrt(X.shape[1]))

    # sample 100 image, reshape, reorg it
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
#%%
X, y = load_data('ex3data1.mat')
plot_100_image(X)
plt.show()
#%%
X_raw, y_raw = load_data('ex4data1.mat', transpose=False)
X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)#增加全部为1的一列
X.shape
def expand_y(y):
#     """expand 5000*1 into 5000*10
#     where y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]: ndarray
#     """
    res = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1

        res.append(y_array)

    return np.array(res)
y = expand_y(y_raw)
#%%
def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']

t1, t2 = load_weight('ex4weights.mat')
t1.shape, t2.shape
#%%
def Feed_forward(theta1,theta2, X):
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    return z2, a2, z3, a3

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

_,_,_,h = Feed_forward(t1, t2, X)
#%%
def cost_function(theta1, theta2, X, y ):
    m = X.shape[0]
    _,_,_,h = Feed_forward(theta1, theta2, X)
    cost = -np.multiply(y, np.log(h)) - np.multiply(1 - y, np.log(1 - h))
    return cost.sum() / m
#%%
cost = cost_function(t1, t2, X, y)

def cost_regularize(theta1, theta2, X, y, lamda = 1):
    cost = cost_function(theta1, theta2, X, y)
    m = X.shape[0]
    sum1 = lamda / 2 / m * np.power(theta1[:, 1:], 2).sum()
    sum2 = lamda / 2 / m * np.power(theta2[:, 1:], 2).sum()
    return sum1 + sum2 + cost

cost_regularize = cost_regularize(t1, t2, X, y, lamda = 1)
#%%
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def initial_theta(shape0, shape1, e = np.power(0.12, 2)):
    theta = []
    for i in range(shape0):
        theta.append(np.random.uniform(-e, e, shape1))
    return np.array(theta)

theta1 = initial_theta(t1.shape[0], t1.shape[1])
theta2 = initial_theta(t2.shape[0], t2.shape[1])
#%%
def gradient(X, y, theta1, theta2):
    z2, a2, z3, a3 = Feed_forward(theta1, theta2, X)
    m = X.shape[0]
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    for i in range(m):
        a1i = X[i, :]  # 1*401
        a3i = a3[i, :] # 1*10
        a2i = a2[i, :] # 1*26
        yi = y[i, :]   # 1*10
        z2i = z2[i, :] # 1*25
        exci3 = yi - a3i # 1*10
        temp = exci3 @ theta2 # 1*26
        temp = temp[1:]
        exci2 = temp * sigmoid_gradient(z2i)# 1*25
        delta1 = delta1 + np.matrix(exci2).T @ np.matrix(a1i)
        delta2 = delta2 + np.matrix(exci3).T @ np.matrix(a2i)
    delta1 = delta1 / m
    delta2 = delta2 / m
    return delta1, delta2
#%%
def gradient_checking():
    np.linalg.
     