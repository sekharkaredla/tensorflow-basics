import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def read_data():
    data_set = pd.read_csv("IRIS.csv",header= None)
    # divide dependent and independent variables
    X = data_set[data_set.columns[0:8]].values
    Y = data_set[data_set.columns[8]]
    # the dependent varibles are being converted to labels
    labelEncoder = LabelEncoder()
    labelEncoder.fit(Y)
    Y = labelEncoder.transform(Y)
    Y = one_hot_encode(Y)
    return (X,Y)

def one_hot_encode(labels):
    # if it is 'setosa' it will be represented as [1, 0, 0] in one_hot_encoder. At a time only one output is hot
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[range(n_labels),labels] = 1
    return one_hot_encode


# getting the data ready
X, Y = read_data()
X, Y = shuffle(X, Y, random_state = 1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

# tensorflow parameters
learning_rate = 0.2
training_epochs = 800
no_of_features = X.shape[1]
no_of_outputs = Y.shape[1]

no_of_nodes_in_hidden_layer_1 = 30
no_of_nodes_in_hidden_layer_2 = 30


# creating dynamic inputs and outputs
inputs = tf.placeholder(dtype=tf.float32, shape=[None,no_of_features])
expected_outputs = tf.placeholder(dtype=tf.float32, shape=[None,no_of_outputs])

weights = {
    "level1": tf.Variable(tf.truncated_normal([no_of_features,no_of_nodes_in_hidden_layer_1])),
    "level2": tf.Variable(tf.truncated_normal([no_of_nodes_in_hidden_layer_1,no_of_nodes_in_hidden_layer_2])),
    "out": tf.Variable(tf.truncated_normal([no_of_nodes_in_hidden_layer_2,no_of_outputs]))
}

baises = {
    "level1": tf.Variable(tf.truncated_normal([no_of_nodes_in_hidden_layer_1])),
    "level2": tf.Variable(tf.truncated_normal([no_of_nodes_in_hidden_layer_2])),
    "out": tf.Variable(tf.truncated_normal([no_of_outputs]))
}


# creating saver for our model
saver = tf.train.Saver()

# creating neural net model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activations
    layer_1 = tf.add(tf.matmul(x, weights['level1']), biases['level1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with sigmoid activations
    layer_2 = tf.add(tf.matmul(layer_1, weights['level2']), biases['level2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with linear activations
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


predicted_outputs = multilayer_perceptron(inputs, weights, baises)

#creating cost function to reduce to mean cross entropy for optimization
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= predicted_outputs, labels= expected_outputs))
#creating step function to generate each step to optimize the above cost function
training_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost_function)

# creating a global initializer here
init = tf.global_variables_initializer()

# creating tensorflow session
sess = tf.Session()
sess.run(init)
# remember in each session the variables will remain the same

# writing model to file
fileWriter = tf.summary.FileWriter(".",sess.graph)


# in the below loop, first 3 lines we are training the neural net using train set and then testing the accuracy using test set in remaining steps
for each_epoch in range(training_epochs):
    sess.run(training_step,feed_dict={inputs:train_x,expected_outputs:train_y})
    cost = sess.run(cost_function,feed_dict={inputs:train_x, expected_outputs:train_y})
    correct_predictions = tf.equal(tf.argmax(predicted_outputs,1),tf.argmax(expected_outputs,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    pred_outputs = sess.run(predicted_outputs,{inputs:test_x})
    mean_square_error = tf.reduce_mean(tf.square(pred_outputs - test_y))
    mean_square_error_value = sess.run(mean_square_error)
    accuracy_value = sess.run(accuracy,{inputs:test_x, expected_outputs: test_y})
    print "epoch number : "+ str(each_epoch) + " - cost : "+ str(cost)+ " - mse : "+ str(mean_square_error_value)+" - accuracy : "+str(accuracy_value)


save_path = saver.save(sess, "/Users/roshni/Documents/CODE/NewProject/TensorFlow Basics/IrisModel")



sess.close()