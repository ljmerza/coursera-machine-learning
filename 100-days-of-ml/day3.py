import pandas as pd
import numpy as np

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values

# encoding data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [3])
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
X = onehotencoder.fit_transform(X).toarray()

# avoid dummy variable - getting rid of one gender
X = X[: , 1:]

# split dataset for training/testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# use multiple linear regression to fit model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predict
y_pred = regressor.predict(X_test)
import matplotlib.pyplot as plt
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , y_pred, color ='green')
plt.show()