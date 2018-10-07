# refer https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/

import tensorflow as tf
from cnn import CNN
from flatten_network import flatten
from fully_connected_layer import create_fully_connected_layer
from batch_dataset import BatchDataset
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
FILTERS = 3
NO_OF_FLOWERS = 5
EPOCHS = 500


input = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT,IMAGE_WIDTH,FILTERS])
output_true = tf.placeholder(tf.float32, shape=[None, NO_OF_FLOWERS])



layer1 = CNN(input=input, num_of_input_channels=3, conv_filter_size=8, num_of_filters=3).cnn
layer2 = CNN(input=layer1, num_of_input_channels=3, conv_filter_size=8, num_of_filters=3).cnn
layer3 = CNN(input=layer2, num_of_input_channels=3, conv_filter_size= 8, num_of_filters= 3).cnn
layer4 = CNN(input=layer3, num_of_input_channels=3, conv_filter_size= 8, num_of_filters= 3).cnn

layer_flat = flatten(layer4)

layer_fully_connected = create_fully_connected_layer(input=layer_flat,no_of_inputs=layer_flat.get_shape()[1:4].num_elements(),no_of_outputs=5)

# fileWriter = tf.summary.FileWriter(".",sess.graph)
print(layer_fully_connected)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= layer_fully_connected, labels= output_true))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost_function)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batchDataset = BatchDataset(0.30)
print "dataset content details:"
batchDataset.get_details()

# for each_epoch in range(EPOCHS):
#     sess.run(optimizer, feed_dict={input: data_train, output_true: labels_train})
#     cost = sess.run(cost_function,feed_dict={input:data_train, output_true:labels_train})
#     correct_predictions = tf.equal(tf.argmax(layer_fully_connected,1),tf.argmax(output_true,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
#     pred_outputs = sess.run(layer_fully_connected,{input:data_test})
#     mean_square_error = tf.reduce_mean(tf.square(pred_outputs - labels_test))
#     mean_square_error_value = sess.run(mean_square_error)
#     accuracy_value = sess.run(accuracy, {input: data_test, output_true: labels_test})
#     print "epoch number : "+ str(each_epoch) + " - cost : "+ str(cost)+ " - mse : "+ str(mean_square_error_value)+" - accuracy : "+str(accuracy_value)


for each_epoch in range(EPOCHS):
    batchDataset.data_present = True
    batchDataset.counter = 0
    data_test, labels_test = batchDataset.get_test_data()
    while(batchDataset.data_present):
        data_train, labels_train = batchDataset.get_next()
        sess.run(optimizer, feed_dict={input: data_train, output_true: labels_train})
        cost = sess.run(cost_function,feed_dict={input:data_train, output_true:labels_train})
        correct_predictions = tf.equal(tf.argmax(layer_fully_connected,1),tf.argmax(output_true,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        pred_outputs = sess.run(layer_fully_connected,{input:data_test})
        mean_square_error = tf.reduce_mean(tf.square(pred_outputs - labels_test))
        mean_square_error_value = sess.run(mean_square_error)
        accuracy_value = sess.run(accuracy, {input: data_test, output_true: labels_test})
        print "epoch number : "+ str(each_epoch) + " - cost : "+ str(cost)+ " - mse : "+ str(mean_square_error_value)+" - accuracy : "+str(accuracy_value)


saver = tf.train.Saver()
save_path = saver.save(sess, "/home/sekhar/EXTRAS/tensorflow-basics/Flower-Dataset")
sess.close()
