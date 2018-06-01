# Euclidean Distance 
# play with k to see how the accuracy differ



import numpy as np
from math import sqrt 
from collections import Counter
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    
    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))

    vote_result = Counter(votes).most_common(1)[0][0]
    # add confidence
    confidence = Counter(votes).most_common(1)[0][1] / k

    # print(vote_result, confidence)

    return vote_result, confidence

# data, add outlier for NAs, drop un-needed columns
df = pd.read_csv('data/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# make sure everything is float -- just in case 
full_data = df.astype(float).values.tolist()

# print first 10 rows 
# print(full_data[:10])

# shuffle the data with random
random.shuffle(full_data)

# slice the data for test and train
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

# populate the dicts
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct +=1

        else:
            print(confidence)
        total +=1

print('Accuracy: ', correct/total)






