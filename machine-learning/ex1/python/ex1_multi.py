#!/usr/bin/env python

# python adaptation of solved ex1_multi.m
# 
# Linear regression with multiple variables
# 
# depends on 
#   - featureNormalize.py
#   - gradientDescentMulti.py
#   - normalEqn.py
#

## Initialization
import numpy as np 
import matplotlib.pyplot as plt
import featureNormalize as fn
import gradientDescentMulti as gdm
import normalEqn as ne

## ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

## Load Data
data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:,:2]
y = data[:,2]
m = len(y) # number of training examples

# Scale features and set them to zero mean
print('Normalizing Features...')
X_norm, mu, sigma = fn.featureNormalize(X)

X_padded = np.column_stack((np.ones((m,1)), X_norm)) # Add a column of ones to x


## ================ Part 2: Gradient Descent ================
print('Running gradient descent...')

alpha = 0.01
num_iters = 10000

# Init Theta and Run Gradient Descent 
theta = np.zeros((3, 1)) 
theta, J_history = gdm.gradientDescentMulti(X_padded, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(J_history.size), J_history, "-b", linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: ')
print("{:f}, {:f}, {:f}\n".format(theta[0,0], theta[1,0], theta[2,0]))

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
area_norm = (1650 - float(mu[:,0])) / float(sigma[:,0])
br_norm = (3 - float(mu[:,1]))/float(sigma[:,1])
house_norm_padded = np.array([1, area_norm, br_norm])

price = np.array(house_norm_padded).dot(theta)

# ============================================================
print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${:,.2f}".format(price[0]))

## ================ Part 3: Normal Equations ================
print('Solving with normal equations...')

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form 
#               solution for linear regression using the normal
#               equations. You should complete the code in 
#               normalEqn.m
#
#               After doing so, you should complete this code 
#               to predict the price of a 1650 sq-ft, 3 br house.
#

## Load Data
data = np.loadtxt('ex1data2.txt', delimiter=",")
X = data[:,:2]
y = data[:,2]
m = len(y) # number of training examples

# Add intercept term to X
X_padded = np.column_stack((np.ones((m,1)), X)) 

# Calculate the parameters from the normal equation
theta = ne.normalEqn(X_padded, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print("{:f}, {:f}, {:f}\n".format(theta[0], theta[1], theta[2]))


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
house_norm_padded = np.array([1, 1650, 3])
price = np.array(house_norm_padded).dot(theta)

# ============================================================

print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${:,.2f}".format(price))