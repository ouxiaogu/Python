import tensorflow as tf
import numpy as np
import glob
import os
import cv2
#import matplotlib.pyplot as plt
#%matplotlib inline
import sys
sys.path.append('/gpfs/DEV/FEM/SHARED/MXP_ModelDB/MXP_toolbox/cnn_sem_model/fromDaniel/SEM')
import img_util
import model_util
import argparse

def load_image(input_image_file_path, SEM_image_file):
    input_img = img_util.pgmp52image(input_image_file_path, scale=False)
    SEM_img = img_util.pgmp52image(SEM_image_file, scale=False)
    print(input_img.shape, SEM_img.shape, input_img.max(), SEM_img.max())
    input_img_list = []
    SEM_img_list = []
    input_img_list.append(input_img)
    SEM_img_list.append(SEM_img)
    input_img_arr = np.expand_dims(np.array(input_img_list), axis=-1)
    SEM_img_arr = np.expand_dims(np.array(SEM_img_list), axis=-1)
    return input_img_arr, SEM_img_arr

def model_apply(model, input_img_arr, truth_img_arr):
    assert os.path.isfile(model), "model not exists!"
    tf.reset_default_graph()
    graph = model_util.load_graph(model)
    x = graph.get_tensor_by_name('input:0')
    y = graph.get_tensor_by_name('truth:0')
    phase = graph.get_tensor_by_name('phase:0')
    pred = graph.get_tensor_by_name('output:0')
    cost = graph.get_tensor_by_name('cost:0')
    in_size = input_img_arr.shape[1] 
    out_size = input_img_arr.shape[1] - 92
    img_size = input_img_arr.shape[1]
    img_center = img_size / 2
    with graph.as_default():
        with tf.Session() as sess:
            X_batch = input_img_arr[:, img_center-in_size/2:img_center+in_size/2, img_center-in_size/2:img_center+in_size/2, :]
            y_batch = truth_img_arr[:, img_center-out_size/2:img_center+out_size/2, img_center-out_size/2:img_center+out_size/2, :]
            cost_val, pred_val = sess.run([cost, pred], feed_dict={x: X_batch, y:y_batch, phase:0})
            #file_writer = tf.summary.FileWriter(logdir='./logs/',graph=sess.graph)
            pred_val = np.squeeze(pred_val, axis=-1)
            return cost_val, pred_val

def model_apply_by_cv(model, config, input_img_arr, truth_img_arr):
    net = cv2.dnn.readNetFromTensorflow(model, config)
    cv_input = input_img_arr.transpose(0,3,1,2)  #feed cv:dnn modulue the order nchw
    cv_truth = truth_img_arr.transpose(0,3,1,2)
    net.setInput(cv_input, 'input:0')
    res = net.forward()
    MSE = ((res - cv_truth) ** 2).mean(axis=None)
    return MSE, res.transpose(0, 2, 3, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model apply')
    parser.add_argument('-i','--input', help='input image', default = "../../data/pattern_1/optical_image.pgm") 
    parser.add_argument('--truth', help='real sem image', default = "../../data/pattern_1/real_sem_image.pgm") 
    parser.add_argument('--i_blob',help='input blob name, default is input', default = "input:0") 
    parser.add_argument('-m', '--model', help='model path', default = "../../data/dlsem_test_model.pb")
    parser.add_argument('--config', help='model graph config path, pbtxt file', default = "../../data/dlsem_test_model.pbtxt")
    parser.add_argument('--o_blob',help='output blob name, default is input', default = "output:0") 
    parser.add_argument('-r','--result', help='result image, default is tfm_result.pgm', default = "tfm_result.pgm") 
    parser.add_argument('--method', help='method for model apply, tf or cv', default = "tf")
    args = vars(parser.parse_args())    
    
    # 20% of the data will automatically be used for validation
    input_img_arr, truth_img_arr = load_image(args["input"], args["truth"])
    if(args["method"] == "tf"):
        cost, pred = model_apply(args["model"], input_img_arr, truth_img_arr)
    elif(args["method"] == "cv"):
        print("use cv dnn module fot tensorflow model apply:\n")
        cost, pred = model_apply_by_cv(args["model"], args["config"], input_img_arr, truth_img_arr)
    print("MSE={}, RMS={}".format(cost, np.sqrt(cost)))
    img_to_save = pred[0, :, :]
    img_util.image2pgm5(img_to_save, args["result"], reverse_scale=False)
