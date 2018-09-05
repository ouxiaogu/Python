import numpy as np
import tensorflow as tf

# generate test data
np.random.seed(6)
x_data = np.random.random([1,100])
y_data = 2 * x_data + 1 + 0.1 * np.random.standard_normal([1,100])

# define the graph
input_holder = tf.placeholder(tf.float32, shape=(1,100))
check_holder = tf.placeholder(tf.float32, shape=(1,100))

w = tf.Variable(tf.random_normal([1], seed = 6))
b = tf.Variable(tf.zeros([1]))
y = w * input_holder + b

loss = tf.reduce_mean(tf.square(y-check_holder));
learning_rate = 0.05
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()

# run the graph
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)

for step in range(1000):
    _, weight, bias = sess.run([train, w, b],
                      feed_dict={input_holder:x_data, check_holder:y_data})
    if step % 100 == 0:
        print(weight, bias)
