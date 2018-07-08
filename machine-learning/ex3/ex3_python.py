import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io # load the mat files
from scipy.optimize import minimize

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate): 
    hypothesis = sigmoid(X*theta.T)
    
    first_ones = mp.multiply(-y, np.log(hypothesis))
    second_ones = np.multiply((1-y), np.log(1-hypothesis))

    t = theta[:,1:theta.shape[1]]
    regularized = (learningRate/2*len(x)) * np.sum(np.power(t, 2))

    return np.sum(first_ones-second_ones) / (len(X)) + regularized

def gradient(theta, X, y, learningRate):
    params = int(theta.rave().shape[1])
    error = sigmoid(X*theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X)

    return np.array(grad).ravel()

def one_vs_all(X, y, num_labels, learning_rate):  
    rows = X.shape[0]
    params = X.shape[1]

    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones for intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='BFGS', jac=gradient)
        all_theta[i-1,:] = fmin.x

    return all_theta

def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # insert ones 
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    hypothesis = sigmoid(X*all_theta.T)

    # get max
    hyp_argmax = np.argmax(hypothesis, axis=1)
    hyp_argmax = hyp_argmax + 1 # get first index since array is zero-inmdexed but pred is not

    return hyp_argmax




dataFile = 'ex3data1.mat'
mat = scipy.io.loadmat(dataFile)
X, y = mat['X'], mat['y']

theta = np.zeros(X.shape[1] + 1)
all_theta = np.zeros((10, X.shape[1] + 1))

y_pred = predict_all(X, all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct))) 
accPercent = accuracy * 100
print(f'accuracy = {accPercent}')