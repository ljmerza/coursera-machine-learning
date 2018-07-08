import numpy as np

def normalEqn(X, y):
    
    theta = np.zeros((X.shape[1], 1))
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    return theta 