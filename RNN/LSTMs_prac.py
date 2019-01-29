import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def read_data():
    data = pd.read_csv("train.csv",header=0)
    X = data[data.columns[:8]].values
    Y = data[data.columns[8:]].values
    return X,Y,data

X,Y,data = read_data()
# X, Y = shuffle(X, Y, random_state = 1)
# train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)


batch_size = 10
no_of_timesteps = 60
no_of_inputs = 7
no_of_outputs = 2
# X_new = np.empty(shape=[batch_size,no_of_timesteps,no_of_inputs],dtype=np.float64)
# def arrange_data(no_of_inputs,no_of_timesteps,batch_size):
#     batch_array = np.empty(shape=[no_of_timesteps,no_of_inputs],dtype=np.float64)
#     count = 0
#     for index,row in data.iterrows():
#         if(count>=batch_size):
#             np.put(X_new,[index],batch_array)
#             batch_array = np.empty(shape=[no_of_timesteps,no_of_inputs])
#             count = 0
#             continue
#         single_input = np.array(row[1:1+no_of_inputs],dtype=np.float64)
#         np.put(batch_array,[count],single_input)
#         count += 1

def arrange_data(no_of_inputs,no_of_timesteps,no_of_outputs):
    arranged_data_X = []
    arranged_data_Y = []
    batch_array_X = np.zeros(shape=[no_of_timesteps, no_of_inputs], dtype=np.float64)
    batch_array_Y = np.zeros(shape=[no_of_timesteps, no_of_outputs], dtype=np.float64)
    count = 0
    for index, row in data.iterrows():
        if(count == no_of_timesteps):
            arranged_data_X.append(batch_array_X)
            arranged_data_Y.append(batch_array_Y)
            batch_array_X = np.zeros(shape=[no_of_timesteps, no_of_inputs], dtype=np.float64)
            batch_array_Y = np.zeros(shape=[no_of_timesteps, no_of_outputs], dtype=np.float64)
            count = 0
        batch_array_X[count] = np.array(row[1:1+no_of_inputs],dtype=np.float64)
        batch_array_Y[count] = np.array(row[1+no_of_inputs:],dtype=np.float64)
        count += 1
    if(count != 0):
        arranged_data_X.append(batch_array_X[:count])
        arranged_data_Y.append(batch_array_Y[:count])
    return arranged_data_X,arranged_data_Y

X_new,Y_new = arrange_data(no_of_inputs,no_of_timesteps,no_of_outputs)
print(X_new,Y_new,len(X_new),len(Y_new))
print(X_new[-1].shape) #according to data should return (24,7)  52 * 50 + 24 = 2624(data size)


