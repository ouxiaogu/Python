import numpy as np
import tensorflow as tf

KSIZE = 11
def gaussian_filter(sigma):
    assert(sigma < KSIZE/6)
    # ksize = int(np.ceil(6*sigma))
    # ksize = ksize+1 if ksize%==2 else ksize
    hlFltSz = KSIZE//2
    a = 1. / (math.sqrt(2*math.pi) * sigma)
    flt_G = np.asarray(list(map(lambda x: a * math.exp(- (x - hlFltSz)**2 / (2.*sigma**2) ), range(0, 2*hlFltSz+1) ) ))
    ss = np.sum(flt_G)
    dst = flt_G/ss
    return dst

# generate test data
np.random.seed(6)
sigma = 1.5
flt_G = gaussian_filter(sigma)
x_data = np.random.random([1,100])
x_blur = np.convolve(x_data[0, :], flt_G, "same").reshape((1, 100))
y_data = 2 * x_blur + 1 + 0.1 * np.random.standard_normal([1,100])

# define the graph
w = tf.Variable(tf.random_normal([1], seed = 6))
s = tf.Variable(tf.random_normal([1], seed = 7))
flt_G_idx = tf.constant(np.arange(KSIZE))
G = 1. / (math.sqrt(2*math.pi) * s) tf.exp(- (flt_G_idx - s)**2 / (2 * s**2))

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





