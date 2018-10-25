import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC






def read_data():
    data_set = pd.read_csv("heart.dat",header= None,delim_whitespace=True)
    # divide dependent and independent variables
    X = data_set[data_set.columns[0:13]].values
    Y = data_set[data_set.columns[13]].values
    # the dependent varibles are being converted to labels
    return (X,Y)



X, Y = read_data()
X, Y = shuffle(X, Y, random_state = 1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)


clf = SVC()
clf.fit(train_x,train_y)


count = 0
i = 0
for each in test_x:
    temp = clf.predict(each.reshape(1,-1))
    if temp == test_y[i]:
        count += 1
    i+=1
print ("accuracy : ",float(count)/len(test_x))