# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-08-31 10:53:39

CNN SEM model Job flow

Last Modified by:  ouxiaogu
"""

import time
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
from simpleNN import SimpleNN

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../libs/tacx")
from MxpJob import MxpJob
from MxpStage import MxpStage, MXP_XML_TAGS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../libs/common")
from logger import logger
from XmlUtil import getConfigData, getGlobalConfigData
log = logger.getLogger(__name__)

###############################################################################
# DLSEMJob Stage Register Area

STAGE_REGISTER_TABLE = {
    'init': 'InitStage',
    'DLSEMCalibration': 'CSemCalStage',
    'DLSEMApply': 'CSemApplyStage'
}
###############################################################################


USAGE_TYPES = ['training', 'validation', 'test']
PATTERN_DEFAULT_ATTR =\
'''name costwt    usage   srcfile    tgtfile  imgpixel   offset_x    offset_y
1   1 training   1_se.bmp    1_image.bmp    1   0   0'''


class DLSEMJob(MxpJob):
    """
    DLSEMJob: Deep learning SEM model job
    """
    def run(self):
        allstages = self.getAllMxpStages(enabled_only=True)
        gcf = self.mxproot.find('.global')
        for stage in allstages:
            stagename, enablenum = stage
            log.info("Stage %s%d starts\n" % (stagename, enablenum))
            cf = self.getStageConfig(stage)
            stagestr = '{}{}'.format(stagename, enablenum)
            curstage = eval(STAGE_REGISTER_TABLE[stagename])(gcf, cf, stagestr, self.jobpath) # MxpStage
            curstage.run()
            outxmlfile = self.getStageIOFile(stage, option=MXP_XML_TAGS[1])
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
        divide_rule = list(map(double, divide_rule.split(":")))

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
        divides = InitStage.divideSample(nsamples, divide_rule)
        log.debug("divides: {}".format(divides))
        usages = df.loc[:, 'usage'].values
        usages[:divides[0]] = USAGE_TYPES[0]
        usages[divides[0]:divides[1]] = USAGE_TYPES[1]
        usages[divides[1]:] = USAGE_TYPES[2]
        df = df.assign(usage=usages)

        # restore df
        self.d_df_patterns = df

    @staticmethod
    def divideSample(nsamples, divide_rule):
        totSlices = sum(divide_rule)
        divides = [max(0, int(d/totSlices*nsamples)) for d in divide_rule]
        while sum(divides) < nsamples:
            gap = nsamples - sum(divides)
            curGap = gap
            for i, d in enumerate(divide_rule):
                if sum(divides) >= nsamples:
                    break
                increment = min(curGap, int(gap * d/totSlices + 0.5))
                divides[i] += increment
                curGap -= increment
        assert(all(np.array(divides)>=0))
        divides = np.cumsum(divides)
        assert(divides[-1]==nsamples)
        return divides

class CSemCalStage(MxpStage):
    def loadDataSet(self):
        nsamples = len(self.d_df_patterns)
        imgsize = getGlobalConfigData(self.d_gcf, self.d_cf, ".imgsize", 128)

        shape_order = getConfigData(self.d_cf, ".shape_order", "nchw")
        self.imgsize = imgsize
        if shape_order == "nchw":
            self.datashape = (nsamples, 1, self.imgsize, self.imgsize)
        else: # "nhwc"
            self.datashape = (nsamples, self.imgsize, self.imgsize, 1)

        train_flt = (self.d_df_patterns.usage==USAGE_TYPES[0])
        train_input_images_file_name = self.d_df_patterns.loc[train_flt, 'srcfile'].values
        train_target_images_file_name = self.d_df_patterns.loc[train_flt, 'tgtfile'].values
        train_input_images = np.array([load_image(path, self.imgsize) for path in train_input_images_file_name])
        train_target_images = np.array([load_image(path, self.imgsize) for path in train_target_images_file_name])

        valid_flt = (self.d_df_patterns.usage==USAGE_TYPES[1])
        valid_input_images_file_name = self.d_df_patterns.loc[valid_flt, 'srcfile'].values
        valid_target_images_file_name = self.d_df_patterns.loc[valid_flt, 'tgtfile'].values
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

        clstype = getGlobalConfigData(self.d_gcf, self.d_cf, 'modeltype', 'simpleNN')
        clstype = SimpleNN if clstype.lower() == 'simplenn' else ConvNN

        cnn = clstype(batchsize=batchsize, epochs=epochs, learning_rate=learning_rate, random_seed=random_seed, imgsize=self.imgsize, data_format = dataformat)
        kwargs = {'stagepath': self.stageresultabspath}
        cnn.train(training_set=(X_train, y_train),
                  validation_set=(X_valid, y_valid), **kwargs)

        # save cnn model
        self.modelpath= os.path.join(self.stageresultabspath, 'tflayers-model')
        cnn.save(path=self.modelpath, epoch=epochs)

        # compute error for all train/validation data
        rmses = []
        input_flt = (self.d_df_patterns.usage!=USAGE_TYPES[2])
        input_srcfiles = self.d_df_patterns.loc[input_flt, 'srcfile'].values
        input_tgtfiles = self.d_df_patterns.loc[input_flt, 'tgtfile'].values
        for i in range(len(input_srcfiles)):
            imgs = [[load_image(input_srcfiles[i], self.imgsize)], [load_image(input_tgtfiles[i], self.imgsize)]]
            rmses.append(cnn.model_error(*imgs))
        self.d_df_patterns.loc[input_flt, 'rms'] = rmses

    def save(self, path, viaDf=True): # override the base save() method, to save via DataFrame
        extraNodes = {}
        extraNodes['model'] = self.modelpath
        super(ContourSelCalStage, self).save(path, viaDf=True, extraNodes=extraNodes)

class CSemApplyStage(MxpStage):
    def loadDataSet(self):
        imgsize = getGlobalConfigData(self.d_gcf, self.d_cf, ".imgsize", 128)
        self.imgsize = imgsize
        test_flt = (self.d_df_patterns.usage==USAGE_TYPES[2])
        if len(test_flt.nonzero()[0]) == 0:
            frac = 0.2
            nsamples = len(self.d_df_patterns)
            testnum = max(1, frac*nsamples)
            log.warning("{} not any pattern labeled as 'test' in the dataset, randomly select data ~frac={}".format(self.stagename, frac))
            test_flt = self.d_df_patterns.usage.isin(USAGE_TYPES).sample(n=testnum, replace=False, random_state=128)
            test_flt = sorted(test_flt.index.values)
        self.test_flt = test_flt
        test_srcfiles = self.d_df_patterns.loc[test_flt, 'srcfile'].values
        test_tgtfiles = self.d_df_patterns.loc[test_flt, 'tgtfile'].values
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
        self.d_df_patterns.loc[self.test_flt, 'applyfile'] = self.d_df_patterns.loc[self.test_flt, 'name'].apply(lambda x: str(x)+ '_CSemApply.jpg')
        y_pred_filenames = self.d_df_patterns.loc[self.test_flt, 'applyfile'].values
        applypath = self.stageresultabspath

        clstype = getGlobalConfigData(self.d_gcf, self.d_cf, 'modeltype', 'simpleNN')
        clstype = SimpleNN if clstype.lower() == 'simplenn' else ConvNN

        cnn = clstype(random_seed=random_seed,imgsize=self.imgsize, data_format=dataformat)
        modelpath = getConfigData(self.d_icf, '.model')
        if not os.path.exists(modelpath):
            raise ValueError("%s, Error, does not exists model path: %s" % (self.stagename, modelpath))
        cnn.load(epoch=epoch, path=modelpath)
        cnn.model_apply(test_X, y_pred_filenames, path=applypath)