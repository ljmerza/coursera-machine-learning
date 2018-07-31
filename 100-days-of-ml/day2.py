import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print('import data')
dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

print('split training data')
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)

print('create linear model object')
# create a regressor to fit data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

print('predict result')
Y_pred = regressor.predict(X_test)

print('visualize training/test results')
labels = {'xlabel':'Hours', 'ylabel':'Score'}
fig, (ax1, ax2) = plt.subplots(2, 1)
plt.subplots_adjust(hspace=0.7)

ax1.set(title='Training Results', **labels)
ax1.scatter(X_train , Y_train, color = 'red')
ax1.plot(X_train , regressor.predict(X_train), color ='blue')

ax2.set(title='Test Results', **labels)
ax2.scatter(X_train , Y_train, color = 'red')
ax2.plot(X_test , regressor.predict(X_test), color ='green')

plt.show()