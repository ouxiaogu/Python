# check tensorboard in http://127.0.0.1:6006
import tensorflow as tf
from tensorboard import main as tb
tf.flags.FLAGS.logdir = "log"
tb.main()