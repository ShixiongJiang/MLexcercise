# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 09:34:42 2021

@author: Shixiong Jiang
"""
#%%
import pandas as pd #read file
import seaborn as sns #plot
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
#%%
df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
df.head()
df.info()
sns.lmplot('population', 'profit', df, size=6, fit_reg=False)
plt.show()
#%%
def get_X(df):
    ones = pd.DataFrame({'ones':np.ones(len(df))})
    data = pd.concat([ones, df], axis=1)
    return data.iloc[:,:-1].values

def get_Y(df):
    return np.array(df.iloc[:, -1])#df.iloc[:, -1] refers to the last column of df

def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean())/ column.std())
#%%
def linear_regression(X_data, y_data, alpha, epoch, optimizer=tf.optimizers.SGD):# 这个函数是旧金山的一个大神Lucas Shen写的
      # placeholder for graph input
    X = tf.placeholder(tf.float32, shape=X_data.shape)
    y = tf.placeholder(tf.float32, shape=y_data.shape)
    
    # construct the graph
    with tf.variable_scope('linear-regression'):
        W = tf.get_variable("weights",
                            (X_data.shape[1], 1),
                            initializer=tf.constant_initializer())  # n*1
    
        y_pred = tf.matmul(X, W)  # m*n @ n*1 -> m*1  X * W

        loss = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)  # (m*1).T @ m*1 = 1*1

    opt = optimizer(learning_rate=alpha)
    opt_operation = opt.minimize(loss)

    # run the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_data = []

        for i in range(epoch):
            _, loss_val, W_val = sess.run([opt_operation, loss, W], feed_dict={X: X_data, y: y_data})
            loss_data.append(loss_val[0, 0])  # because every loss_val is 1*1 ndarray

            if len(loss_data) > 1 and np.abs(loss_data[-1] - loss_data[-2]) < 10 ** -9:  # early break when it's converged
                # print('Converged at epoch {}'.format(i))
                break

    # clear the graph
    tf.reset_default_graph()
    return {'loss': loss_data, 'parameters': W_val}  # just want to return in row vector format
#%%   
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])#读取数据，并赋予列名

data.head()
X = get_X(data)
print(X.shape, type(X))

y = get_Y(data)
print(y.shape, type(y))

theta = np.zeros(X.shape[1])
def lr_cost(theta, X, y):
    m = X.shape[0]#the number of samples
    inner = X @ theta - y
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost

cost = lr_cost(theta, X, y)

def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)
    return inner / m

def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()
    
    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))
        
    return _theta, cost_data

#%%
epoch = 1000
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)
#%%
final_theta
#%%
a=[]
for i in range(epoch+1):
    a.append(i)
ax = sns.lineplot(a, cost_data)
ax.set_xlabel('epoch')
ax.set_ylabel('cost')
plt.show()

a = final_theta[0]
b = final_theta[1]
plt.scatter(df.population, df.profit, label="Training data")
plt.plot(data.population, data.population*b + a, label="prediction")
plt.show()
#%%
def gradient_decent_for_plot(theta, X, y, epoch, alpha=0.01):
    cost_data = [lr_cost(theta, X, y)]
    thetak = theta.copy()
    _theta = []
    _theta.append(theta)
    for _ in range(epoch):
        thetak = thetak - alpha * gradient(thetak, X, y)
        _theta.append(thetak)
        cost_data.append(lr_cost(thetak, X, y))
    
    return _theta, cost_data

#%%
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])#读取数据，并赋予列名

data.head()
X = get_X(data)
print(X.shape, type(X))

y = get_Y(data)
print(y.shape, type(y))

theta = np.zeros(X.shape[1])
epoch = 5000
theta_list, cost_data_list = gradient_decent_for_plot(theta, X, y, epoch)
x_line = []
y_line = []
for i in range(epoch + 1):
    x_line.append(theta_list[i][0])
    y_line.append(theta_list[i][1])
z_line = cost_data_list
fig = plt.figure(figsize = (10, 7))

 
# Creating plot
ax.contour(x_line, y_line, z_line, color = "green")
plt.title("simple 3D scatter plot")


plt.show()