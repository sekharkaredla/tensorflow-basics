import tensorflow as tf
import numpy as np

n_inputs = 4 # number of inputs
n_neurons = 7 # number of units in hidden in layer
n_timesteps = 2 # no of steps going back


X_batch = np.array([
        [[0, 1, 2, 5], [9, 8, 7, 4]], # Batch 1
        [[3, 4, 5, 2], [0, 0, 0, 0]], # Batch 2
        [[6, 7, 8, 5], [6, 5, 4, 2]], # Batch 3
    ])

print(X_batch.shape)


X = tf.placeholder(tf.float32, [None, n_timesteps, n_inputs]) # placeholder for input

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) # basic RNN cell

outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

print(outputs,states)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch}) # evaluates outputs (values of hidden layers) in each step. Output dimensions are [batch_size, max_time, cell.output_size]
    states_val = states.eval(feed_dict={X: X_batch}) # Returns the final state of the network. Output dimensions are [batch_size, cell.state_size]

print(outputs_val,outputs_val.shape)
print(states_val,states_val.shape)