import numpy as np

def computeCost(X, y, theta):
    m = y.size
    J = 0
    
    h = hypothesis(X, theta)
    sqError = np.power(h-y, 2)

    J = 1/(2*m) * np.sum(sqError)
    
    return J

def hypothesis(X, theta):
    return X.dot(theta)