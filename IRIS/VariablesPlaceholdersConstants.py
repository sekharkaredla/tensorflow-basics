import tensorflow as tf

W = tf.Variable([-1.5],dtype= tf.double)
b = tf.Variable([1.5],dtype= tf.double)

x = tf.placeholder(dtype=tf.double)
y = tf.placeholder(dtype=tf.double)
linear_model = W * x + b

session = tf.Session()

initializer = tf.global_variables_initializer()

squared_delta = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_delta)

session.run(initializer)


print session.run(loss, {x:[1,2,3,4],y:[0,-1,-2,-3]})

fileWriter = tf.summary.FileWriter(".",session.graph)

session.close()