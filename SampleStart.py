import tensorflow as tf
const1 = tf.constant(5.0)
const2 = tf.constant(2.0)

print const1, const2

const3 = const1 * const2

session = tf.Session()

fileWriter = tf.summary.FileWriter(".",session.graph)

print session.run(const3)

session.close()
