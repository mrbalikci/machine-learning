# pickling and scaling: serilization of any python object 
# you may use AWS for a short period of time. 
# save your data into .pickle and 
# run it on your local machine 

# pip install sklearn
# pip install quandl
# pip install pandas

# take the data and 
# figure out the best fit line for the data
# equation of the line 
# y = mx + b

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style 
import pickle 

# open pickle save it and read it 

style.use('ggplot')

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


# print(df.head(5))

###
# Train and Test 
### 

# define X and scale it 

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)

X = X[:-forecast_out]
X_lately = X[-forecast_out:]

# skale X -- normalized -1 to 1


# redefine X 
df.dropna(inplace=True)
y = np.array(df['label'])

# print(len(X), len(y))

# train and test data, 20% for test data 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)

# clasifiers
# n_jobs means number of jobs, -1 means as many as 
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)

# ### 
# # saving clasifier / pickling 
# ### 

# # not to have to re-train the data
# # imagine if the data is very large
# # you may train it once in a while 

# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

# test the accuracy of the test data 
accuracy = clf.score(X_test, y_test)

# print(accuracy)

# forecast set
# prediction with sklearn
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

###
# Plot the data and the date
###

df['Forecast'] = np.nan

# the last day
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# populate df with the new dates
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# Plot the data
df['Adj. Close'].plot()
df['Forecast'].plot()

plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





