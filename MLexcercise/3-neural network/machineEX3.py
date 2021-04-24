# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 22:14:00 2021

@author: Shixiong Jiang
neural network
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
X, y = load_data('ex3data1.mat')

print(X.shape)
print(y.shape)
def plot_an_image(image):
#     """
#     image : (400,)
#     """
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))
    
    
#%%
pick_one = np.random.randint(0, 5000)
plot_an_image(X[pick_one, :])
plt.show()
print('this should be {}'.format(y[pick_one]))
#%%
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

plot_100_image(X)
plt.show()
#%%
raw_X, raw_y = load_data('ex3data1.mat')
X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1)#插入了第一列（全部为1）
X.shape

y_matrix = []
for k in range(1, 11):
    y_matrix.append((raw_y == k).astype(int)) 
y_matrix = [y_matrix[-1]] + y_matrix[:-1]
y = np.array(y_matrix)
y.shape
#%%
def regularize_cost(theta, x, y, l = 1):
    cost = cost_function_logistic(theta, x, y)
    theta_j1_to_n = theta[1:]
    return cost + l * 1 / (2 * len(x)) * np.power(theta_j1_to_n, 2).sum()
def cost_function_logistic(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def regularize_gradient(theta, x, y, l=1):
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n
    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradient_decent_logistic(theta, X, y) + regularized_term
def gradient_decent_logistic(theta, X, y):
    m = len(y)
    return X.T @(sigmoid(X @ theta) - y) / m
def predict(theta, X):
    ans = sigmoid(X @ theta)
    return (ans > 0.5).astype(int)

def logistic_regression(X, y, lamda = 1):
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularize_cost, x0=theta,args=(X, y, lamda), method='TNC', jac=regularize_gradient, options={'disp': True})
    final_theta = res.x
    return final_theta
t0 = logistic_regression(X, y[0])
y_pred = predict(t0, X)
print('accuracy{}'.format(np.mean(y_pred == y[0])))
#%%
accuracy = 0
for k in range(10):
    t0 = logistic_regression(X, y[k])
    y_pred = predict(t0, X)
    accuracy = np.mean(y[k] == y_pred)
    

#%%

X, y = load_data('ex3data1.mat',transpose=False)

X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1) 
def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']
theta1, theta2 = load_weight('ex3weights.mat')
z1 = X @ theta1.T
temp = sigmoid(z1)
a2 = np.insert(temp, 0, values=np.ones(temp.shape[0]), axis=1)
z3 = a2 @ theta2.T
out = sigmoid(z3)
y_pred = np.argmax(out, axis=1) + 1 
print(classification_report(y, y_pred))