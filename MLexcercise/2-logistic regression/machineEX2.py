# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:29:23 2021

@author: Shixiong 
logistic regression
"""
#%%
import pandas as pd #read file
import seaborn as sns #plot
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
#%%
df = pd.read_csv('ex2data1.txt', names=['Exam1', 'Exam2','class'])
df.head()
df.info()
df.describe()
sns.lmplot('Exam1', 'Exam2',hue='class',data=df, size=6, fit_reg=False)
plt.show()
#%%
def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data = pd.concat([ones, df], axis=1)
    return data.iloc[:, :-1].values

def get_y(df):
    return df.iloc[:, -1]
def nolmalize_feature(df):
    return df.apply(lambda column : (column - column.mean()) / column.std())
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
#%%
X = get_X(df)
y = get_y(df)
theta = np.zeros(3)
plt.plot(np.arange(-10, 10,step=0.1), sigmoid(np.arange(-10,10, step=0.1)))
plt.xlabel('z')
plt.ylabel('sigmoid')
plt.title('sigmoid function')
plt.show()
#%%

def cost_function_logistic(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

cost = cost_function_logistic(theta, X, y)

def gradient_decent_logistic(theta, X, y):
    m = len(y)
    return X.T @(sigmoid(X @ theta) - y) / m

gradient = gradient_decent_logistic(theta, X, y)

    
#%% find the final theta
import scipy.optimize as opt
res = opt.minimize(fun=cost_function_logistic, x0=theta, args=(X, y), method='Newton-CG', jac=gradient_decent_logistic)
print(res)

def predict(theta, X):
    ans = sigmoid(X @ theta)
    return (ans > 0.5).astype(int)
final_theta = res.x
y_pred = predict(final_theta, X)
print(classification_report(y, y_pred))
#%%plot the graph for classfication
df.describe()
x_line = np.arange(20,120, step=0.5)
k = final_theta / final_theta[2] * -1
y_line = k[0] + k[1] * x_line
sns.lmplot('Exam1', 'Exam2',hue='class',data=df, size=6, fit_reg=False)
plt.plot(x_line, y_line)
plt.show()
#%%The second data set
df = pd.read_csv('ex2data2.txt', names=['Test1', 'Test2', 'class'])
sns.lmplot('Test1', 'Test2', hue='class', data = df,size = 6,fit_reg=False)
plt.show()
#%%
def mapfeature(x, y, power, as_ndarray=False):
    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
            for i in np.arange(power + 1)
            for p in np.arange(i + 1)}
    return pd.DataFrame(data).values

#%%
x1 = np.array(df.Test1)
x2 = np.array(df.Test2)
X = mapfeature(x1, x2, 6)
theta = np.zeros((X.shape[1]))
y = get_y(df)

def regularize_cost(theta, x, y, l = 1):
    cost = cost_function_logistic(theta, x, y)
    theta_j1_to_n = theta[1:]
    return cost + l * 1 / (2 * len(x)) * np.power(theta_j1_to_n, 2).sum()

regularize_init_cost = regularize_cost(theta, X, y)

def regularize_gradient(theta, x, y, l=1):
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient_decent_logistic(theta, X, y) + regularized_term
#%%
import scipy.optimize as opt
res = opt.minimize(fun=regularize_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularize_gradient)
fianl_theta = res.x
y_pred = predict(fianl_theta, X)
print(classification_report(y_pred, y))
#%%
    
def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = mapfeature(x_cord, y_cord, power)  # this is a dataframe

    inner_product = mapped_cord @ theta

    decision = mapped_cord[np.abs(inner_product) < threshhold]

    return decision[:,1], decision[:,2]
def draw_line(power, l):
    df = pd.read_csv('ex2data2.txt', names=['Test1', 'Test2', 'class'])
    x1 = np.array(df.Test1)
    x2 = np.array(df.Test2)
    X = mapfeature(x1, x2, power)
    y = get_y(df)
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularize_cost, x0=theta, args=(X, y,l), method='Newton-CG', jac=regularize_gradient)
    final_theta = res.x
    density = 1000
    threshhold = 2 * 10**-3
    m, n = find_decision_boundary(density, power, final_theta, threshhold)
    
    sns.lmplot('Test1', 'Test2', hue='class', data=df, size=6, fit_reg=False, scatter_kws={"s": 100})
    plt.scatter(m, n, s=10)
    plt.title('Decision boundary, lamda={}'.format(l))
    plt.show()
#%%
draw_line(6, 0)
draw_line(6, 1)
draw_line(6,10)
draw_line(6, 100)


