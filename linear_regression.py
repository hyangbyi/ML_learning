# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:21:49 2017

@author: Austin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

import seaborn as sns
    
# 计算损失函数
def computeCost(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0
    
    h = X.dot(theta)
    
    J = 1.0/(2*m)*(np.sum(np.square(h-y)))
    
    return J

# 梯度下降
def gradientDescent(X, y, theta=[[0],[0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    
    for iter in np.arange(num_iters):
        #线性决策函数
        h = X.dot(theta)
        #梯度求最佳theta
        theta = theta - alpha*(1.0/m)*(X.T.dot(h-y))
        #代价函数，最小处即是最佳theta
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)
    
def main():
    data = np.loadtxt("linear_regression_data1.txt", delimiter=',')
    
    X = np.c_[np.ones(data.shape[0]), data[:, 0]] #两列，一列全1， 一列data的第一列
    y = np.c_[data[:, 1]]   #一列，data的第二列

    # plt.scatter(X[:, 1], y, s=30, c='r', marker='x', linewidths=1)
    # plt.xlim(4, 24)
    # plt.xlabel('Population of City in 10,000s')
    # plt.ylabel('Profit in $10,000s');

    J = computeCost(X, y)
    print(J)

    theta, cost_J = gradientDescent(X, y)
    print(theta.ravel())

    plt.plot(cost_J)
    plt.ylabel('Cost J')
    plt.xlabel('Iterations');

    xx = np.arange(5, 23)
    yy = theta[0]+theta[1]*xx
    
    plt.scatter(X[:, 1], y, s=30, c='r', marker='x', linewidths=1)
    plt.plot(xx,yy, label='Linear regression (Gradient descent)')
    
    regr = LinearRegression()
    regr.fit(X[:,1].reshape(-1, 1), y.ravel())
    plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')
    
    plt.xlim(4,24)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.legend(loc=4)
    
    print(theta.T.dot([1, 3.5])*10000)
    print(theta.T.dot([1, 7])*10000)
    
if __name__ == '__main__':
    main()

