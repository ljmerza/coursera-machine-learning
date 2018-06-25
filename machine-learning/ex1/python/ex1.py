#/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

from warmUpExercise import warmUpExercise
from computeCost import computeCost
from plotData import plotData
from gradientDescent import gradientDescent 

# Machine Learning Online Class - Exercise 1: Linear Regression
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#
# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
warmUpExercise()

print('Program paused. Press enter to continue.\n')
input()
 
# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
X, y = np.genfromtxt('ex1data1.txt', delimiter=',', unpack =True)

# https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.c_.html
# Translates slice objects to concatenation along the second axis
X_ones = np.ones(X.shape[0])
X = np.c_[X_ones, X]
y = np.c_[y]

# # Plot Data
# # Note: You have to complete the code in plotData.m
plotData(X, y)

# =================== Part 3: Cost and Gradient descent ===================
theta = np.zeros((X.shape[1], 1))

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = {0:2.2f}'.format(J))
print('Expected cost value (approx) 32.07\n')

# # further testing of the cost function
J = computeCost(X, y, [[-1], [2]])
print('With theta = [-1 ; 2]\nCost computed = {0:2.2f}'.format(J))
print('Expected cost value (approx) 54.24')

print('Program paused. Press enter to continue.\n')
input()

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, Cost_J = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('theta: ',theta.ravel())
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')

# Predict values for population sizes of 35,000 and 70,000
predict1 = theta.T.dot([1, 3.5]) * 10000
print('For population = 35,000, we predict a profit of #f\n', predict1)
predict2 = theta.T.dot([1, 7]) * 10000
print('For population = 70,000, we predict a profit of #f\n', predict2)

print('Program paused. Press enter to continue.\n')
input()

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Create grid coordinates for plotting
B0 = np.linspace(-10, 10, 50) # Return evenly spaced numbers over a specified interval
B1 = np.linspace(-1, 4, 50)  # (from -1 to 4 return 50 numbers evenly spaced)

#Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields 
#over N-D grids, given one-dimensional coordinate arrays x1, x2,..., xn
xx, yy = np.meshgrid(B0, B1, indexing='xy')
Z = np.zeros((B0.size,B1.size)) # make zero matrix same size

# Calculate Z-values (Cost) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z): #Return an iterator yielding pairs of array coordinates and values. 
    Z[i,j] = computeCost(X,y, theta=[[xx[i,j]], [yy[i,j]]])


fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot

CS = ax1.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap=plt.jet)
ax1.scatter(theta[0],theta[1], c='r')

# Right plot
ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(Z.min(),Z.max())
ax2.view_init(elev=15, azim=230)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)

plt.show()
