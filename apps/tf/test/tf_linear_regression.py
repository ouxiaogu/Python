import numpy as np
import tensorflow as tf

# generate test data
np.random.seed(6)
x_data = np.random.random([1,100])
y_data = 2 * x_data + 1 + 0.1 * np.random.standard_normal([1,100])

# define the graph
w = tf.Variable(tf.random_normal([1], seed = 6))
b = tf.Variable(tf.zeros([1]))
y = w * x_data + b

loss = tf.reduce_mean(tf.square(y-y_data));
learning_rate = 0.05
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()

# run the graph
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)

for step in range(1000):
    sess.run(train)
    if step % 100 == 0:
        print(sess.run(w), sess.run(b))





