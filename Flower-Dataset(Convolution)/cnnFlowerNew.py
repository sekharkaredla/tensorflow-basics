#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


# In[3]:


learning_rate = 0.0001
epochs = 10
batch_size = 50

inputs = tf.placeholder(tf.float32, [None, 3072])
inputs_reshaped = tf.reshape(inputs, [-1, 32, 32, 3])
outputs_expected = tf.placeholder(tf.float32, [None, 5])


# In[4]:


def one_hot_encode(labels):
    # if it is 'label1' it will be represented as [1, 0, 0] in one_hot_encoder. At a time only one output is hot
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[range(n_labels),labels] = 1
    one_hot_encode.astype(float)
    return one_hot_encode

def split_dataset_with_ratio(ratio):
    data = []
    labels = []
    flower_data_folder = "/Users/roshni/Documents/CODE/flowers_dataset/reshaped/"
    for each_flower_image in glob.glob(flower_data_folder+"*.jpg"):
        image_data = cv2.imread(each_flower_image)
        image_data = image_data.astype(float)
        image_label = each_flower_image.split('/')[-1].split('_')[0]

        image_data = (image_data - image_data.min())/(image_data.max() - image_data.min())
        image_data = image_data.reshape(-1)
        data.append(image_data)
        labels.append(image_label)
        print image_label
        print image_data

    data, labels = shuffle(data, labels, random_state=1)
    labelEncoder = LabelEncoder()
    labelEncoder.fit(labels)
    labels = labelEncoder.transform(labels)
    labels = one_hot_encode(labels=labels)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=ratio, random_state=42)
    return data_train, labels_train, data_test, labels_test


# In[5]:


class BatchDataset:
    def __init__(self,ratio,batch_size):
        data = split_dataset_with_ratio(ratio)
        self.data_train = data[0]
        self.labels_train = data[1]
        self.data_test = data[2]
        self.labels_test = data[3]
        self.counter = 0
        self.data_present = True
        self.batch_size = batch_size

    def get_next(self):
        if self.counter+self.batch_size<len(self.data_train):
            self.counter += self.batch_size
            return (self.data_train[self.counter:self.counter+self.batch_size],self.labels_train[self.counter:self.counter+self.batch_size])
        else:
            self.data_present = False
            return (self.data_train[self.counter:],self.labels_train[self.counter:])

    def get_test_data(self):
        return (self.data_test,self.labels_test)

    def get_details(self):
        print len(self.data_train), len(self.labels_train), len(self.data_test), len(self.labels_test)


# In[6]:


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

   
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

  
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')


    out_layer += bias

    
    out_layer = tf.nn.relu(out_layer)


    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


# In[7]:


layer1 = create_new_conv_layer(inputs_reshaped, 3, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')

# In[8]:


flattened_layer = tf.reshape(layer2, [-1, 8 * 8 * 64])
weight_flat = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1200], stddev=0.03), name='weight_flat')
bais_flat = tf.Variable(tf.truncated_normal([1200], stddev=0.01), name='bais_flat')
dense_layer1 = tf.matmul(flattened_layer, weight_flat) + bais_flat
dense_layer1 = tf.nn.relu(dense_layer1)


# In[9]:


weight_final_softmax = tf.Variable(tf.truncated_normal([1200, 5], stddev=0.03), name='weight_final_softmax')
bais_final_softmax = tf.Variable(tf.truncated_normal([5], stddev=0.01), name='bais_final_softmax')
dense_layer2 = tf.matmul(dense_layer1, weight_final_softmax) + bais_final_softmax
outputs_predicted = tf.nn.softmax(dense_layer2)


# In[10]:


cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=outputs_expected))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_function)
correct_prediction = tf.equal(tf.argmax(outputs_expected, 1), tf.argmax(outputs_predicted, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

# In[ ]:

batchDataset = BatchDataset(0.30,4)
print "dataset content details:"
batchDataset.get_details()
sess = tf.Session()
sess.run(init)


for each_epoch in range(epochs):
    batchDataset.data_present = True
    batchDataset.counter = 0
    data_test, labels_test = batchDataset.get_test_data()
    while(batchDataset.data_present):
        data_train, labels_train = batchDataset.get_next()
        sess.run(optimizer, feed_dict={inputs: data_train, outputs_expected: labels_train})
    cost = sess.run(cost_function,feed_dict={inputs:data_train, outputs_expected:labels_train})
    pred_outputs = sess.run(outputs_predicted,{inputs:data_test})
    mean_square_error = tf.reduce_mean(tf.square(pred_outputs - labels_test))
    mean_square_error_value = sess.run(mean_square_error)
    accuracy_value = sess.run(accuracy, {inputs: data_test, outputs_expected: labels_test})
    print "epoch number : "+ str(each_epoch) + " - cost : "+ str(cost)+" - mse : "+str(mean_square_error_value)+" - accuracy : "+str(accuracy_value)
    


# In[ ]:




