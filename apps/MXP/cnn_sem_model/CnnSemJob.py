# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-08-31 10:53:39

CNN SEM model Job flow

Last Modified by: ouxiaogu
"""

import time
import xml.etree.ElementTree as ET
import re
from collections import OrderedDict
import numpy as np
import glob
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import pandas as pd
from dataset import DataSet, load_image, centered_norm
from convNN import ConvNN
import argparse

import os, os.path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../tacx")
from MxpJob import MxpJob
from MxpStage import *
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../common")
from XmlUtil import setConfigData
import logger

log = logger.setup("MxpJob", "debug")


###############################################################################
# MXP DL SEM model Job Stage Register Area

STAGE_REGISTER_TABLE = {
    'DLSEMInit': 'DLSemInitStage',
    'DLSEMCalibration': 'CSemCalStage',
    'DLSEMApply': 'CSemApplyStage'
}
###############################################################################

USAGE_TYPES = ['training', 'validation', 'test']
PATTERN_DEFAULT_ATTR =\
'''name costwt    usage   srcfile    tgtfile  imgpixel   offset_x    offset_y
1   1 training   1_se.bmp    1_image.bmp    1   0   0'''

class CnnSemJob(MxpJob):
    """docstring for MxpJob"""
    def run(self):
        allstages = self.getAllMxpStages()
        gcf = self.mxproot.find('.global')
        for stage in allstages:
            stagename, enablenum = stage
            log.info("Stage %s%d starts\n" % (stagename, enablenum))
            df = None
            if stagename != "init":
                inxmlfile = self.getStageIOFile(stage, option=MXP_XML_TAGS[0])
                stagexml = MxpStageXmlParser(inxmlfile, option=MXP_XML_TAGS[0])
                df = stagexml.geticcfs()
            cf = self.getStageConfig(stage)
            stagepath = self.resultAbsPath('{}{}'.format(stagename, enablenum))
            curstage = eval(STAGE_REGISTER_TABLE[stagename])(gcf, cf, df, stagename, self.jobpath)
            curstage.run()
            outxmlfile = self.getStageIOFile(stage, option="outxml")
            curstage.save(outxmlfile)
            log.info("Stage %s%d successfully finished\n" % (stagename, enablenum))

class InitStage(MxpStage):
    def run(self):
        dataDir = getConfigData(self.d_cf, "data_dir", "")
        folderflt = getConfigData(self.d_cf, ".filter/folder", "*")
        srcfileflt = getConfigData(self.d_cf, ".filter/srcfile", "*")
        tgtfileflt = getConfigData(self.d_cf, ".filter/tgtfile", "*")
        doshuffle = getConfigData(self.d_cf, ".shuffle", 0)
        divide_rule = getConfigData(self.d_cf, ".divide_rule", "70:20:10")
        divide_rule = list(map(int, divide_rule.split(":")))
        if sum(divide_rule) != 100:
            log.warn("Warning, sum of training set, validation set and test set should be 100, input is %s, use [70, 20, 10] instead." % ':'.join(divide_rule))

        if not os.path.exists(dataDir):
            raise IOError("data_dir not exists at: {}".format(dataDir))
        log.info('Going to glob dataset')
        log.info('Now going to read path for both input and target images')
        srcpathex = os.path.join(dataDir, folderflt, srcfileflt)
        tgtpathex = os.path.join(dataDir, folderflt, tgtfileflt)
        srcfiles = glob.glob(srcpathex)
        tgtfiles = glob.glob(tgtpathex)
        if len(srcfiles) != len(tgtfiles):
            raise ValueError("%s, number of source files is not equal with target files', %d != %d\n", len(srcfiles), len(tgtfiles))
        nsamples = len(srcfiles)
        basename = list(map(os.path.basename, (map(os.path.dirname, srcfiles))))
        DATA = StringIO(PATTERN_DEFAULT_ATTR)
        defaultdf = pd.read_csv(DATA, sep="\s+")
        colnames = defaultdf.columns
        defaultrow = defaultdf.as_matrix()
        dataset = np.tile(defaultrow, (nsamples, 1))
        df = pd.DataFrame(dataset, columns=colnames)
        df = df.assign(name=basename, srcfile=srcfiles, tgtfile=tgtfiles)
        if doshuffle > 0:
            df = df.sample(frac=1).reset_index(drop=True) # shuffle

        # assign usage
        divides = [max(1, int(d/100.*nsamples)) for d in divide_rule]
        divides[-1] = nsamples - sum(divides[:-1])
        assert(all(np.array(divides)>=0))
        divides = np.cumsum(divides)
        log.debug("divides: {}".format(divides))
        usages = df.loc[:, 'usage'].values
        usages[:divides[0]] = USAGE_TYPES[0]
        usages[divides[0]:divides[1]] = USAGE_TYPES[1]
        usages[divides[1]:] = USAGE_TYPES[2]
        df = df.assign(usage=usages)

        # restore df
        self.d_df = df

class CSemCalStage(MxpStage):
    def loadDataSet(self):
        nsamples = len(self.d_df)
        imgsize = getGlobalConfigData(self.d_gcf, self.d_cf, ".imgsize", 128)

        shape_order = getConfigData(self.d_cf, ".shape_order", "nchw")
        self.imgsize = imgsize
        if shape_order == "nchw":
            self.datashape = (nsamples, 1, self.imgsize, self.imgsize)
        else: # "nhwc"
            self.datashape = (nsamples, self.imgsize, self.imgsize, 1)

        train_flt = (self.d_df.usage==USAGE_TYPES[0])
        train_input_images_file_name = self.d_df.loc[train_flt, 'srcfile'].values
        train_target_images_file_name = self.d_df.loc[train_flt, 'tgtfile'].values
        train_input_images = np.array([load_image(path, self.imgsize) for path in train_input_images_file_name])
        train_target_images = np.array([load_image(path, self.imgsize) for path in train_target_images_file_name])

        valid_flt = (self.d_df.usage==USAGE_TYPES[1])
        valid_input_images_file_name = self.d_df.loc[valid_flt, 'srcfile'].values
        valid_target_images_file_name = self.d_df.loc[valid_flt, 'tgtfile'].values
        valid_input_images = np.array([load_image(path, self.imgsize) for path in valid_input_images_file_name])
        valid_target_images = np.array([load_image(path, self.imgsize) for path in valid_target_images_file_name])

        doCenteredNorm = getGlobalConfigData(self.d_gcf, self.d_cf, 'centered_normalize_X', 0)
        if doCenteredNorm > 0:
            centered_norm(train_input_images)
            centered_norm(valid_input_images)
        log.debug("X train shape {} {}".format(train_input_images.shape, train_input_images[0].shape))
        self.train = DataSet(train_input_images, train_target_images, train_input_images_file_name, train_target_images_file_name)
        self.valid = DataSet(valid_input_images, valid_target_images, valid_input_images_file_name, valid_target_images_file_name)

        log.info("Complete reading input data")
        log.info("Number of files in Training-set:\t\t{}".format(len(train_input_images_file_name)))
        log.info("Number of files in Validation-set:\t{}".format(len(valid_input_images_file_name)))

    def run(self):
        learning_rate = getConfigData(self.d_cf, ".learning_rate", 1e-4)
        batchsize = getConfigData(self.d_cf, ".batchsize", 8)
        epochs = getConfigData(self.d_cf, ".epochs", 2)
        random_seed = getGlobalConfigData(self.d_gcf, self.d_cf, '.random_seed', 123)
        device = getGlobalConfigData(self.d_gcf, self.d_cf, '.device', 'cpu')
        dataformat = 'channels_last' if device.lower() == 'cpu' else 'channels_first'

        self.loadDataSet()
        X_train, y_train = self.train.input_images, self.train.target_images
        X_valid, y_valid = self.valid.input_images, self.valid.target_images
        log.info('Training:\t{}\t{}'.format(str(X_train.shape), str(y_train.shape)))
        log.info('Validation:\t{}\t{}'.format(str(X_valid.shape), str(y_valid.shape)))
        cnn = ConvNN(batchsize=batchsize, epochs=epochs, learning_rate=learning_rate, random_seed=123, imgsize=self.imgsize, data_format = dataformat)
        kwargs = {'stagepath': self.stagepath}
        cnn.train(training_set=(X_train, y_train),
                  validation_set=(X_valid, y_valid), **kwargs)

        # save cnn model
        self.modelpath= os.path.join(self.stagepath, 'tflayers-model')
        cnn.save(path=self.modelpath, epoch=epochs)

        # compute error for all train/validation data
        rmses = []
        input_flt = (self.d_df.usage!=USAGE_TYPES[2])
        input_srcfiles = self.d_df.loc[input_flt, 'srcfile'].values
        input_tgtfiles = self.d_df.loc[input_flt, 'tgtfile'].values
        for i in range(len(input_srcfiles)):
            imgs = [[load_image(input_srcfiles[i], self.imgsize)], [load_image(input_tgtfiles[i], self.imgsize)]]
            rmses.append(cnn.model_error(*imgs))
        self.d_df.loc[input_flt, 'rms'] = rmses

    def save(self, path):
        ocf = dfToMxpOcf(self.d_df)
        log.debug("Result save path: %s\n" % (path))
        setConfigData(ocf, 'model', self.modelpath)
        tree = ET.ElementTree(ocf)
        tree.write(path, encoding="utf-8", xml_declaration=True)
        log.debug("Result saved at %s\n" % (path))

class CSemApplyStage(MxpStage):
    def loadDataSet(self):
        imgsize = getGlobalConfigData(self.d_gcf, self.d_cf, ".imgsize", 128)
        self.imgsize = imgsize
        test_flt = (self.d_df.usage==USAGE_TYPES[2])
        if len(test_flt.nonzero()[0]) == 0:
            frac = 0.2
            nsamples = len(self.d_df)
            testnum = max(1, frac*nsamples)
            log.warning("{} not any pattern labeled as 'test' in the dataset, randomly select data ~frac={}".format(self.stagename, frac))
            test_flt = self.d_df.usage.isin(USAGE_TYPES).sample(n=testnum, replace=False, random_state=128)
            test_flt = sorted(test_flt.index.values)
        self.test_flt = test_flt
        test_srcfiles = self.d_df.loc[test_flt, 'srcfile'].values
        test_tgtfiles = self.d_df.loc[test_flt, 'tgtfile'].values
        test_X = np.array([load_image(path, self.imgsize) for path in test_srcfiles])
        test_y = np.array([load_image(path, self.imgsize) for path in test_tgtfiles])
        doCenteredNorm = getGlobalConfigData(self.d_gcf, self.d_cf, 'centered_normalize_X', 0)
        if doCenteredNorm > 0:
            centered_norm(test_X)
        self.test = DataSet(test_X, test_y, test_srcfiles, test_tgtfiles)

    def run(self):
        device = getGlobalConfigData(self.d_gcf, self.d_cf, '.device', 'cpu')
        dataformat = 'channels_last' if device.lower() == 'cpu' else 'channels_first'
        random_seed = getGlobalConfigData(self.d_gcf, self.d_cf, '.random_seed', 123)
        epoch = getConfigData(self.d_cf, ".epoch", 2)

        self.loadDataSet()

        test_X = self.test.input_images
        self.d_df.loc[self.test_flt, 'applyfile'] = self.d_df.loc[self.test_flt, 'name'].apply(lambda x: str(x)+ '_CSemApply.jpg')
        y_pred_filenames = self.d_df.loc[self.test_flt, 'applyfile'].values
        applypath = self.stagepath
        cnn = ConvNN(random_seed=random_seed,imgsize=self.imgsize, data_format=dataformat)
        inxmlfile = getConfigData(self.d_cf, MXP_XML_TAGS[0])
        inxmlfile = os.path.join(self.jobresultpath, inxmlfile)
        parser = MxpStageXmlParser(inxmlfile, option=MXP_XML_TAGS[0])
        modelpath = getConfigData(parser.geticf(), '.model')
        if not os.path.exists(modelpath):
            raise ValueError("%s, Error, does not exists model path: %s" % (self.stagename, modelpath))
        cnn.load(epoch=epoch, path=modelpath)
        cnn.model_apply(test_X, y_pred_filenames, path=applypath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xml-driving CNN sem job')
    parser.add_argument('jobpath', help='cnn sem job path')
    args = parser.parse_args()
    jobpath = args.jobpath
    if jobpath is None:
        parser.print_help()
        parser.exit()
        # jobpath = './samplejob'
        # print('No jobpath is inputed, use sample job path: %s' % jobpath)
    print(str(vars(args)))
    myjob = CnnSemJob(jobpath)
    myjob.run()