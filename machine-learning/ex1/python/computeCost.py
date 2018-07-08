import numpy as np

def computeCost(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)
    return (1.0/(2*m)) * errorSquared(h, y).sum(axis=0)

def hypothesis(X, theta):
    return X.dot(theta)

def errorSquared(h, y):
    error_val =  h - np.transpose([y])
    return  np.power(error_val, 2)
    