# pip install sklearn
# pip install quandl
# pip install pandas

# take the data and 
# figure out the best fit line for the data
# equation of the line 
# y = mx + b

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression 

# the data 
df = quandl.get('WIKI/GOOGL')

# select needed columns 
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# define new column
# high - low difference percentage 
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100.0

# percent change 
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# print(df.head(10))

forecast_col = 'Adj. Close'

# fill NAs with an outlier 
df.fillna(-99999, inplace = True)

# math.ceil gets round
forecast_out = int(math.ceil(0.01*len(df)))
# print(forecast_out)

# need labels , adjusted closed price  
df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)
# print(df.head(5))

###
# Train and Test 
### 

# define X and y 

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# skale X -- normalized -1 to 1
X = preprocessing.scale(X)

# redefine X 
df.dropna(inplace=True)
y = np.array(df['label'])

# print(len(X), len(y))

# train and test data, 20% for test data 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)

# clasifiers 
# n_jobs means number of jobs, -1 means as many as 
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# test the accuracy of the test data 
accuracy = clf.score(X_test, y_test)

print(accuracy)

###
# SVM - Support Vector Machine 
### 
clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)

# test the accuracy of the test data 
accuracy_svm = clf.score(X_test, y_test)

print(accuracy_svm)

