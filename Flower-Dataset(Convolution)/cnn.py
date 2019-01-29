import tensorflow as tf
from weights import Weight
from biases import Bais
class CNN:
    def __init__(self, input, num_of_input_channels, conv_filter_size, num_of_filters):
        self.weights = Weight(shape= [conv_filter_size, conv_filter_size, num_of_input_channels, num_of_filters])
        self.baises = Bais(size=num_of_filters)


        self.cnn = tf.nn.conv2d(input= input, filter= self.weights.weights, strides= [1, 1, 1, 1], padding= 'SAME')

        self.cnn += self.baises.biases

        self.cnn = tf.nn.max_pool(value= self.cnn, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')

        self.cnn = tf.nn.relu(self.cnn)

