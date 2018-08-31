from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
normal_factor = 65535
debug = 1


def load_train(train_path, imgsize, number, input_tag, target_tag):
    '''
    # @brief load training image folder into array of np.arrays 
    # @param train_path: the training images folder 
    # @param imgsize: the imgsize of each images read into array of np.array. Image must be larger than this size
    # @param input_tag: regex keyword for input image file name: image full name will be "<patternid><input_tag>". Default is "_optical_image.pgm" 
    # @param target_tag: regex keyword for output image file name. default is "sem_image.pgm"
    # @param number: the total number of patterns loaded into result. default value -1 mean load all patterns
    #
    # @return: input_images (array of np.array), target_images, input_images_file_name, target_images_file_name 
    '''
    assert(imgsize > 0), "imgsize of load_train function is <= 0!"
    input_images = []
    target_images = []
    input_images_file_name = []
    target_images_file_name = []

    print('Going to read training images')
    print('Now going to read input image files and target image')
    path = os.path.join(train_path, "*" + input_tag)
    files = glob.glob(path)
    if(number > 0):
        files = files[:number]
    assert(len(files) > 0), "empty input_images on image folders\n"
    for fl in files:
        if(debug == 1):
            print("read input image %s\n" % fl)
        image = cv2.imread(fl, -1)
        image = cv2.resize(image, (imgsize, imgsize), 0,0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / normal_factor)
        np.reshape(image, (1, imgsize,imgsize))
        input_images.append(image)
        flbase = os.path.basename(fl)
        input_images_file_name.append(flbase)
        #read target images        
        fl_target = fl.replace(input_tag, target_tag)       
        if(debug == 1):
            print("read target image %s\n" % fl_target)
        t_image = cv2.imread(fl_target, -1)
        t_image = cv2.resize(t_image, (imgsize, imgsize), 0,0, cv2.INTER_LINEAR)
        t_image = t_image.astype(np.float32)
        t_image = np.multiply(t_image, 1.0 / normal_factor)
        np.reshape(t_image, (1,imgsize,imgsize)) #NCHW order
        target_images.append(t_image)
        fl_target_base = os.path.basename(fl_target)
        target_images_file_name.append(fl_target_base)

    num_img = len(input_images)
    input_images = np.array(input_images)
    input_images = np.reshape(input_images,(num_img, 1, imgsize, imgsize))
    num_target = len(target_images)
    target_images = np.array(target_images)
    target_images = np.reshape(target_images, (num_target, 1, imgsize, imgsize))
    input_images_file_name = np.array(input_images_file_name)
    target_images_file_name = np.array(target_images_file_name)
    return input_images, target_images, input_images_file_name, target_images_file_name

def load_image(filepath, imgsize = 512):
    '''
    load a single sem image into np.array
    ''' 
    image = cv2.imread(filepath, -1)
    image = cv2.resize(image, (imgsize, imgsize), 0,0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / normal_factor)
    np.reshape(image, (1, imgsize,imgsize))
    return image


def batch_generator(X, y, batchsize=1,shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]    
    for i in range(0, X.shape[0], batchsize):
        yield (X[i:i+batchsize, :], y[i:i+batchsize])

class DataSet(object):

  def __init__(self, input_images, target_images, input_images_file_name, target_images_file_name):
    self._num_examples = input_images.shape[0]

    self._input_images = input_images
    self._target_images = target_images
    self._input_images_file_name = input_images_file_name
    self._target_images_file_name = target_images_file_name
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def input_images(self):
    return self._input_images

  @property
  def target_images(self):
    return self._target_images

  @property
  def input_images_file_name(self):
    return self._input_images_file_name

  @property
  def target_images_file_name(self):
    return self._target_images_file_name

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._input_images[start:end], self._target_images[start:end], self._input_images_file_name[start:end], self._target_images_file_name[start:end]


def read_train_sets(train_path, imgsize, validation_size, number, input_tag, target_tag):
  class DataSets(object):
    pass
  data_sets = DataSets()

  input_images, target_images, input_images_file_name, target_images_file_name = load_train(train_path, imgsize, number, input_tag, target_tag)
  input_images, target_images, input_images_file_name, target_images_file_name = shuffle(input_images, target_images, input_images_file_name, target_images_file_name)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * input_images.shape[0])

  validation_input_images = input_images[:validation_size]
  validation_target_images = target_images[:validation_size]
  validation_input_images_file_name = input_images_file_name[:validation_size]
  validation_target_images_file_name = target_images_file_name[:validation_size]
  train_input_images = input_images[validation_size:]
  train_target_images = target_images[validation_size:]
  train_input_images_file_name = input_images_file_name[validation_size:]
  train_target_images_file_name = target_images_file_name[validation_size:]
  data_sets.train = DataSet(train_input_images, train_target_images, train_input_images_file_name, train_target_images_file_name)
  data_sets.valid = DataSet(validation_input_images, validation_target_images, validation_input_images_file_name, validation_target_images_file_name)
  return data_sets
