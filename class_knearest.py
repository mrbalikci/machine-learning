# Classification with K-nearest neighbors 
# create model best divide/seperate the data 
# figure out how to seperate them into obvious groups -- clustering 
# create a model define the clustering 
# based on clustering, classify a new data point 
# if you have more than 2d, machine classify 
# k-nearest: k=2 means 2 closest neighbor to the data point 
# Odd k values ideal 
# If 3 groups k 5 is good 
# degree of confidence: 
# larger data set, this model not better 
# support vector machine much better for larger data set 

# takes any new data point and compares euclidian distance and defines the class

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

accuracies = []
for i in range(25):
    df = pd.read_csv('data/breast-cancer-wisconsin.data.txt')

    # replace missing data as an outlier
    # good practice
    # df.dropna(inplace=True)
    df.replace('?', -99999, inplace=True)

    # drop not-needed column(s)
    # to increase the credibility of the model 
    df.drop(['id'], 1, inplace=True)

    # define X and y
    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    # train and test 
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

    # define classifier here

    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)

    # test the accuracy 
    accuracy = clf.score(X_test, y_test)
    print('the accuracy is %s' % accuracy)

    # make a prediction
    # you can pickle the model if the data was a large 
    # make up numbers 
    example_measures = np.array(([4,2,1,1,1,2,3,2,1], [2,3,2,1,4,2,7,2,1]))


    example_measures = example_measures.reshape(len(example_measures),-1)

    prediction = clf.predict(example_measures)
    print(prediction)
    accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies))

