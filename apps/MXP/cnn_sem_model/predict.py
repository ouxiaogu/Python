from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
debug = 0
from dataset import *

if(len(sys.argv) == 3):
    debug = int(sys.argv[2]) 

if(debug == 1):
    import pdb;
    pdb.set_trace();

# First, pass the path of the image
image_path=sys.argv[1] 
image_size=512
num_channels=1
image = load_image(image_path, image_size)
x_batch = image.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('cnn-sem-model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()
'''
print("all tensors in graph to graph_variable_name_list.txt:\n")
f = open("graph_variable_name_list.txt", 'w')
for op in graph.get_operations():
    f.write(str(op.name))
    f.write("\n")
f.close()
'''

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")
y_true = graph.get_tensor_by_name("y_true:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_test_images = np.zeros([1, image_size,image_size,num_channels], dtype=float) 
print("y_test_image shape",y_test_images.shape);

### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
result = np.reshape(result, [image_size, image_size])
imax = np.amax(result)
imin = np.amin(result)
irange = np.maximum(imax-imin, 0.01)
print("(min, max) is ", imin, imax)
result = (result - imin)/irange
cv2.imwrite("result.jpg", result * 255)
np.savetxt("result.txt", result, delimiter=',')
#print(result)
'''
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
