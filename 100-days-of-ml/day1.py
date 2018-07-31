import numpy as np
import pandas as pd

dataset = pd.read_csv('Data.csv')


# pandas.DataFrame.iloc
# Purely integer-location based indexing for selection by position.
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# handle missing data - any missing data (NaN values) is filled in with the
# mean of that column of data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[: 1:3])

# encoding catergorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])

# create dummy variable
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toArray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# split datasets into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.fit_transform(X_test)

