import tensorflow as tf


class Bais:
    def __init__(self,size):
        self.biases = tf.Variable(tf.constant(0.04, shape=[size]))