# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:59:24 2017

@author: Austin
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures

import seaborn as sns

def loadData(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print("dims: ", data.shape)
    print(data[1:6, :])
    return(data)
    
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1
    if axes == None:
        axes = plt.gca()  
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, 
            linewidths=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True)   

#定义sigmoid函数
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))     

#定义损失函数
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    try:
        J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    except Exception(e):
        return(np.inf)    
               
    if np.isnan(J[0]):
        return(np.inf)
    return J[0]

def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    
    grad =(1.0/m)*X.T.dot(h-y)

    return(grad.flatten())

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

def main():
    data = loadData("data1.txt", ",")
    X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
    y = np.c_[data[:, 2]]
    plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
    initial_theta = np.zeros(X.shape[1])
    cost = costFunction(initial_theta, X, y)
    grad = gradient(initial_theta, X, y)
    print('Cost: \n', cost)
    print('Grad: \n', grad)
    res = minimize(costFunction, initial_theta, 
                   args=(X,y), jac=gradient, options={'maxiter':400})
    score = sigmoid(np.array([1, 45, 85]).dot(res.x.T))
    print(score)

    plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
    plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    x2_min, x2_max = X[:,2].min(), X[:,2].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');
    
if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        
        
        
        