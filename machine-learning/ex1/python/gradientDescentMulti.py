import numpy as np
from computeCost import computeCost

def gradientDescentMulti(X, theta_start = np.zeros(2)):
    theta = theta_start
    jvec = [] #Used to plot cost as function of iteration
    thetahistory = [] #Used to visualize the minimization path later on

    for col in np.arange(iterations):
        itr_theta = theta
        cost = computeCost(X, y, theta)
        jvec.append(cost)

        thetahistory.append( list(theta[:,0]) )

        for j in np.arange(len(tmptheta)):
             tmptheta[j] = theta[j] - (alpha/m)*np.sum((h(initial_theta,X) - y)*np.array(X[:,j]).reshape(m,1))


def hypothesis(theta, X):
    return np.dot(X,theta)

def computeCost(theta, X, y):
    h = hypothesis(mytheta,X)

    return float((1./(2*m)) * np.dot((h-y).T, (h-y)))