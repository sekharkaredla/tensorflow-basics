import tensorflow as tf
from cnn import CNN
from flatten_network import flatten
from fully_connected_layer import create_fully_connected_layer
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
FILTERS = 3
NO_OF_FLOWERS = 5

input = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT,IMAGE_WIDTH,FILTERS])
output_true = tf.placeholder(tf.float32, shape=[None, NO_OF_FLOWERS])



layer1 = CNN(input=input, num_of_input_channels=3, conv_filter_size=8, num_of_filters=3).cnn
layer2 = CNN(input=layer1, num_of_input_channels=3, conv_filter_size=8, num_of_filters=3).cnn
layer3 = CNN(input=layer2, num_of_input_channels=3, conv_filter_size= 8, num_of_filters= 3).cnn
layer4 = CNN(input=layer3, num_of_input_channels=3, conv_filter_size= 8, num_of_filters= 3).cnn

layer_flat = flatten(layer4)

layer_fully_connected = create_fully_connected_layer(input=layer_flat,no_of_inputs=layer_flat.get_shape()[1:4].num_elements(),no_of_outputs=5)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
fileWriter = tf.summary.FileWriter(".",sess.graph)
print(layer_fully_connected)
sess.close()