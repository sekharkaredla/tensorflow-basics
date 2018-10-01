import tensorflow as tf

class Weight:
    def __init__(self,shape):
        self.weights = tf.Variable(tf.truncated_normal(shape = shape, stddev=0.04))