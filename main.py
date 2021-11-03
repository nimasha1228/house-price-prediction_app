# Import the Packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Import the data
df = pd.read_csv('data.csv')

# Feature Selection
columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

X = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#  Create Machine Learning Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Dump the model using pickle
pickle.dump(lr, open('model.pkl', 'wb'))


