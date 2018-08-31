from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import dataset
import tensorflow as tf
import convNN
from convNN import *
import time
from datetime import timedelta
import math
import random
import numpy as np
import sys
import argparse
debug = 0 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train CNN SEM model')
    parser.add_argument('-t','--type', help='train, or model_apply', default = "train")
    parser.add_argument('-d','--dir', help='train image folder', default = "./training_data_3") 
    parser.add_argument('--input_tag', help='input image, default is se term image', default = "_seTermComponent.pgm") 
    parser.add_argument('--target_tag', help='target image, default is sem image', default = "_image.pgm") 
    parser.add_argument('--epochs', help='epochs iteration number,default is 20', default = 20) 
    args = vars(parser.parse_args())    
    
    # set debug option
    dataset.debug = debug
    convNN.debug == debug
    
    # 20% of the data will automatically be used for validation
    _validation_size = 0.2
    _batchsize = 4 
    _imgsize = 256 
    _load_imgsize = _imgsize 
    if(args['type'] == "model_apply"):
        _load_imgsize = _imgsize 
    _number = -1
    _input_tag = args["input_tag"]
    _target_tag = args["target_tag"] 
    train_path= args["dir"]
    _epochs = int(args["epochs"])
    # We shall load all the training and validation input_images and target_images into memory using openCV and use that during training
    data = dataset.read_train_sets(train_path, imgsize=_load_imgsize, 
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
    if(debug == 1):
        print("Main: min and max of X_train_centered: %f, %f \n" %(X_train.min(), X_train.max()))
        print("Main: min and max of y_train: %f, %f \n" %(y_train.min(), y_train.max()))
    mean_vals = np.mean(X_train, axis=0) 
    std_val = np.std(X_train)
    if(debug == 1):
        print("std val is: ", std_val)
    X_train_centered = (X_train - mean_vals)/std_val
    X_valid_centered = (X_valid - mean_vals)/std_val
    del X_train, X_valid
    mean_vals = np.mean(y_train, axis=0) 
    std_val = np.std(y_train)
    y_train_centered = (y_train - mean_vals)/std_val
    y_valid_centered = (y_valid - mean_vals)/std_val
    del y_train, y_valid
    
    if(args['type'] == "train"):    
        cnn = ConvNN(batchsize=_batchsize, epochs=_epochs, random_seed=123, imgsize=_imgsize)
        if(debug == 1):
            print("Main: min and max of X_train_centered: %f, %f \n" %(X_train_centered.min(), X_train_centered.max()))
            print("Main: min and max of y_train: %f, %f \n" %(y_train.min(), y_train.max()))
        cnn.train(training_set=(X_train_centered, y_train_centered), 
                  validation_set=(X_valid_centered, y_valid_centered))
        cnn.save(epoch=_epochs)
    elif(args['type'] == "model_apply"):
        cnn = ConvNN(random_seed=123,imgsize=_imgsize)
        cnn.load(epoch=_epochs)
        X_train_file_name = [i.replace(_input_tag, "_dl_simu.pgm") for i in X_train_file_name]
        X_valid_file_name = [i.replace(_input_tag, "_dl_simu.pgm") for i in X_valid_file_name]
        cnn.model_apply(X_train_centered, X_train_file_name)
        cnn.model_apply(X_valid_centered, X_valid_file_name)
    elif(args['type'] == "freeze"):
        cnn = ConvNN(random_seed=123,imgsize=_imgsize)
        cnn.freeze(epoch=_epochs, model_name="dlsem")
        

