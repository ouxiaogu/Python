# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import dataset
import tensorflow as tf
import numpy as np
import os
import cv2

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
debug = 0


class ConvNN(object):
    def __init__(self, batchsize=8,
                 epochs=20, learning_rate=1e-4, 
                 dropout_rate=0.5,
                 shuffle=True, random_seed=None, 
                 imgsize=128):
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        self.imgsize = imgsize
        self.data_format = "channels_last"
                
        g = tf.Graph()
        with g.as_default():
            ## set random-seed:
            tf.set_random_seed(random_seed)
            
            ## build the network:
            self.build()

            ## initializer
            self.init_op = \
                tf.global_variables_initializer()
                
            ## saver
            self.saver = tf.train.Saver()
            
        ## create a session
        self.sess = tf.Session(graph=g)
                
    def build(self):
        
        ## Placeholders for X and y:
        tf_x = tf.placeholder(tf.float32, 
                              shape=[None, 1, self.imgsize, self.imgsize],
                              name='tf_x')
        tf_y = tf.placeholder(tf.float32, 
                              shape=[None, 1, self.imgsize, self.imgsize],
                              name='tf_y')
        is_train = tf.placeholder(tf.bool, 
                              shape=(),
                              name='is_train')
        
        ## 1st layer: Conv_1
        h1_filters_size = 128
        tf_x_shape = tf_x.get_shape().as_list()
        tf_x_nchw = tf.reshape(tf_x, [-1, tf_x_shape[2], tf_x_shape[3], tf_x_shape[1]]) #change to nchw order
        h1 = tf.layers.conv2d(tf_x_nchw, 
                              kernel_size=(5, 5), 
                              filters=h1_filters_size, 
                              activation=tf.nn.relu, 
                              data_format = self.data_format) 
        ## MaxPooling
        h1_pool = tf.layers.max_pooling2d(h1, 
                              pool_size=(2, 2), 
                              strides=(2, 2),
                              data_format = self.data_format)
        ## 2n layer: Conv_2
        h2_filter_size = 128
        h2 = tf.layers.conv2d(h1_pool, kernel_size=(5,5), 
                              filters=64, 
                              activation=tf.nn.relu,
                              data_format = self.data_format)
        ## MaxPooling 
        h2_poo_size = 2
        h2_pool = tf.layers.max_pooling2d(h2, 
                              pool_size=(h2_poo_size, h2_poo_size), 
                              strides=(2, 2),
                              data_format = self.data_format)

        ## 3rd layer: Fully Connected
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool, 
                              shape=[-1, n_input_units])
        h3 = tf.layers.dense(h2_pool_flat, 1024, 
                              activation=tf.nn.relu)

        ## Dropout
        h3_drop = tf.layers.dropout(h3, 
                              rate=self.dropout_rate,
                              training=is_train)
        
        ## 4th layer: Fully Connected (linear activation)
        h4 = tf.layers.dense(h3_drop, self.imgsize * self.imgsize, 
                              activation=None)  
        
        s = tf_y.get_shape().as_list()
        y_pred = tf.reshape(h4, [-1,s[1], s[2], s[3]], name= "y_pred")  
        
        '''            
        h4 = tf.layers.dense(tf_x, self.imgsize * self.imgsize, 
                              activation=None)  
        s = tf_y.get_shape().as_list()
        y_pred = tf.reshape(h4, [-1, s[1], s[2], s[3]], name = "y_pred") 
        '''
 
        ## Predictition
        predictions = {
            'rms': tf.cast(tf.reduce_mean(tf.pow(tf.subtract(y_pred, tf_y), 2)), 
                              tf.float32, name='rms')}
        
        ## Loss Function and Optimization
        rms_loss = tf.reduce_mean(tf.pow(tf.subtract(y_pred, tf_y), 2), name = 'rms_loss')
        
        ## Optimizer:
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(rms_loss, name='train_op')

    def save(self, epoch=20, path='./tflayers-model/'): 
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Saving model in %s' % path)
        self.saver.save(self.sess, 
                        os.path.join(path, 'model.ckpt'),
                        global_step=epoch)
        
        
        
    def load(self, epoch =20, path = './tflayers-model/'):
        print('Loading model from %s' % path)
        self.saver.restore(self.sess, 
             os.path.join(path, 'model.ckpt-%d' % epoch))
    
    def freeze(self, epoch =20, path = './tflayers-model/', model_name="dlsem"):
        print('freeze model from %s' % path)
        self.saver.restore(self.sess, 
             os.path.join(path, 'model.ckpt-%d' % epoch))
        # freeze the graph
        output_node_names = ["y_pred", "rms_loss"]
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,
            self.sess.graph_def,
            output_node_names)
        # Save the frozen graph
        with open(os.path.join(path, model_name + ".pb"), 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
        
    def train(self, training_set, 
              validation_set=None,
              initialize=True):
        
        self.train_cost_ = []
        X_data = np.array(training_set[0])
        
        y_data = np.array(training_set[1])

        ## initialize variables
        if initialize:
            feed = {"tf_x:0": X_data,
                    "tf_y:0": y_data,
                    'is_train:0': True}
            self.sess.run(self.init_op, feed_dict=feed)
            #self.sess.run(tf.global_variables_initializer())


        for epoch in range(1, self.epochs + 1):
            batch_gen = \
               dataset.batch_generator(X_data, y_data, batchsize=self.batchsize, 
                                 shuffle=self.shuffle)
            avg_loss = 0.0
            for i, (batch_x,batch_y) in \
                enumerate(batch_gen):
                #print("batch_x shape: ", batch_x.shape)
                #print("batch_y shape: ", batch_y.shape)
                feed = {'tf_x:0': batch_x, 
                        'tf_y:0': batch_y,
                        'is_train:0': True} ## for dropout
                loss, y_pred, _ = self.sess.run(
                        ['rms_loss:0', 'y_pred:0', 'train_op'], 
                        feed_dict=feed)
                if(debug ==1):
                    print("Train function: min and max of y_pred: %f, %f \n" %(y_pred.min(), y_pred.max()))
                    print("Train function: min and max of y_train: %f, %f \n" %(batch_y.min(), batch_y.max()))
                avg_loss += loss
                
            print('Epoch %02d: Training rms: '
                  '%7.7f' % (epoch, avg_loss), end=' ')
            if validation_set is not None:
                feed = {'tf_x:0': batch_x, 
                        'tf_y:0': batch_y,
                        'is_train:0': False} 
                valid_rms = self.sess.run('rms_loss:0',
                                          feed_dict=feed)
                print('Validation rms: %7.7f' % valid_rms)
            else:
                print()
                    
    def predict(self, X_test):
        feed = {'tf_x:0': X_test,
                'is_train:0': False} 
        return self.sess.run('rms_loss:0',
                                 feed_dict=feed)
        
    def model_apply(self, X_data, X_data_file_name, path='./model-apply/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Saving model apply images in %s' % path)
        
        #reshape the data
        original_shape = X_data.shape
        pad_shape = (np.prod(original_shape) // (self.imgsize * self.imgsize) + 1)*self.imgsize *self.imgsize
        X_data_padding = np.resize(X_data, pad_shape)
        X_data_padding = np.reshape(X_data_padding,[-1, 1, self.imgsize, self.imgsize])
        
        feed = {'tf_x:0': X_data_padding,               
                  'is_train:0':False
        }
        
        y_pred = self.sess.run('y_pred:0', feed_dict=feed)
        print("min and max of y_pred: %f, %f \n" %(y_pred.min(), y_pred.max()))
        count = 0
        images = np.resize(y_pred, np.prod(original_shape)) #remove those padding
        images = images.reshape([-1,original_shape[2], original_shape[3]])
        for image in images:
           scaled_image = 255 * (image - image.min())/(image.max() - image.min())
           name = X_data_file_name[count]
           print("write simulation sem image file to: %s\n" % name)
           cv2.imwrite(os.path.join(path, name), scaled_image)
           count = count + 1
       
            
if __name__ == "__main__":
    
    
    batch_size = 8 
    DIR = ".\\"

    import argparse
    parser = argparse.ArgumentParser(description='demo CNN SEM model')
    parser.add_argument('-t','--type', help='train, or model_apply', required=True)
    args = vars(parser.parse_args())    
    
    # 20% of the data will automatically be used for validation
    _validation_size = 0.2
    _imgsize = 128 
    _number = -1
    _input_tag = "_optical_image.pgm"
    _target_tag = "_sem_image.pgm"
    train_path='training_data'
    # We shall load all the training and validation input_images and target_images into memory using openCV and use that during training
    data = dataset.read_train_sets(train_path, imgsize=_imgsize, 
                  validation_size=_validation_size, number=_number,
                  input_tag=_input_tag, target_tag=_target_tag)

    print("Complete reading input data")
    print("Number of files in Training-set:\t\t{}".format(len(data.train.target_images)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.target_images)))   
    X_train, y_train = data.train.input_images, data.train.target_images
    X_train_file_name, y_train_file_name = data.train.input_images_file_name, data.train.target_images_file_name
    X_valid, y_valid = data.valid.input_images, data.valid.target_images
    X_valid_file_name, y_valid_file_name = data.valid.input_images_file_name, data.valid.target_images_file_name
    print('Training:   ', X_train.shape, y_train.shape)
    print('Validation: ', X_valid.shape, y_valid.shape)    
    mean_vals = np.mean(X_train, axis=0) 
    std_val = np.std(X_train)
    X_train_centered = (X_train - mean_vals)/std_val
    X_valid_centered = X_valid - mean_vals
    del X_train, X_valid
    
    if(args['type'] == "train"):    
        cnn = ConvNN(batchsize=8, random_seed=123, imgsize=_imgsize)
        cnn.train(training_set=(X_train_centered, y_train), 
                  validation_set=(X_valid_centered, y_valid))
        cnn.save(epoch=20)
    elif(args['type'] == "model_apply"):
        cnn = ConvNN(random_seed=123,imgsize=_imgsize)
        cnn.load(epoch=20)
        X_data_file_name = [i.replace(_input_tag, "_dl_simu.pgm") for i in X_train_file_name]
        cnn.model_apply(X_train_centered, X_data_file_name)

