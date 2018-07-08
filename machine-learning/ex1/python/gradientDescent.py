import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta=np.zeros(2), alpha=0.01, iterations=10000):
    m = len(y)
    J_history = np.zeros(iterations)

    for itr in np.arange(iterations):
        hypothesis = X.dot(theta)
        error = hypothesis - np.transpose([y])

        gradient = (1.0/m) * X.T.dot(error)
        theta = theta - alpha * gradient
        
        J_history[itr] = computeCost(X, y, theta)

    return theta, J_history