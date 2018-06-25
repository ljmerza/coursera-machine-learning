import numpy as np
import matplotlib.pyplot as plt

from gradientDescentMulti import gradientDescentMulti 

datafile = 'ex1data2.txt'

cols = np.loadtxt(datafile, delimiter=',', usecols=(0,1,2), unpack=True)

X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))

m = y.size # number of training examples

# insert ones column into X matrix
X = np.insert(X, 0, 1, axis=1)

plt.grid(True)
plt.xlim([-100,5000])
dummy = plt.hist(X[:,0],label = 'col1')
dummy = plt.hist(X[:,1],label = 'col2')
dummy = plt.hist(X[:,2],label = 'col3')
plt.title('Feature Raw Data')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show()

feature_means = []
feature_stds = []

Xnorm = X.copy()
for col in np.arange(Xnorm.shape[1]):
    feature_means.append(np.mean(Xnorm[:,col]))
    feature_stds.append(np.std(Xnorm[:,col]))

    # skip first column when gdoing normalization
    if not col: continue

    # get last mean/std added
    Xnorm[:,col] = (Xnorm[:,col] - feature_means[-1]) / feature_stds[-1]

# feature normalized data
plt.grid(True)
plt.xlim([-5,5])
dummy = plt.hist(Xnorm[:,0],label = 'col1')
dummy = plt.hist(Xnorm[:,1],label = 'col2')
dummy = plt.hist(Xnorm[:,2],label = 'col3')
plt.title('Feature Normalization')
plt.xlabel('Column Value')
plt.ylabel('Counts')
dummy = plt.legend()
plt.show()

initial_theta = np.zeros((Xnorm.shape[1],1))
theta, thetahistory, jvec = gradientDescentMulti(Xnorm, initial_theta)