import tensorflow as tf
from weights import Weight
from biases import Bais

def create_fully_connected_layer(input, no_of_inputs, no_of_outputs):
    weights = Weight(shape=[no_of_inputs, no_of_outputs])
    biases = Bais(no_of_outputs)
    layer = tf.matmul(input, weights.weights) + biases.biases
    return layer


