# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-25 14:53:41

Contour Selection model calibration stage

Last Modified by:  ouxiaogu
"""

import numpy as np
import pandas as pd
import io
import pickle
import time
import multiprocessing

from ContourSelBaseModel import ContourSelBaseModel
from ContourSelClfModel import ContourSelClfModel
from ContourSelRuleModel import ContourSelRuleModel, addNeighborFeatures

import sys
import os, os.path
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../libs/tacx/")
from SEMContourEncrypted import parseContourWrapper
from MxpStage import MxpStage

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../libs/common/")
from XmlUtil import getConfigData
from logger import logger
log = logger.getLogger(__name__)

__all__ = ["ContourSelCalStage"]

USAGE_TYPES = ['training', 'validation', 'test']

class ContourSelCalStage(MxpStage):
    """
    ContourSelCalStage

    Contour Select App: Contour point classification model calibration stage

    Traverse Patterns with cost>0, based on pattern image and contour, labeling
    the Outlier BBox, the contour points inside current bbox will be considered
    'Outlier', add a 'UserLabel' columns as 'bad', others considered as 'good', 
    the labeled data will be used for model calibration/verification, the 
    UserLabel is ground truth.
    """
    allSrcColNames = 'slope, intensity, ridge_intensity, contrast, EigenRatio'
    srcColNames = 'slope, intensity, ridge_intensity, contrast'
    tgtColName = 'UserLabel'
    outColName = 'ClfLabel'
    DEFAULT_ADI_COLS = 'ridge_intensity, slope, intensity, NeighborOrientation, NeighborParalism'
    DEFAULT_AEI_COLS = 'intensity, ridge_intensity, slope, contrast, NeighborParalism'
    neighborColNames = ['NeighborOrientation', 'NeighborParalism'] # used neighbor filters
    allNeighborColNames = ['NeighborContinuity', 'NeighborOrientation', 'NeighborParalism'] 
    debugOn = True

    def __init__(self, gcf, cf, stagename, jobpath):
        super(ContourSelCalStage, self).__init__(gcf, cf, stagename, jobpath)
        self.__getXTrainCols()
        
        self.reuseModel = getConfigData(self.d_cf, '.reuse_model', 1) > 0
        self.useMultiprocess = getConfigData(self.d_cf, '.multiprocess', 1) > 0
        self.applyModel = getConfigData(self.d_cf, '.apply_model', 0) > 0

    def __getXTrainCols(self):
        jobtype = getConfigData(self.d_cf, ".jobtype", 'adi')
        self.AEI = False if jobtype.lower() == 'adi' else True

        default_cols = self.DEFAULT_AEI_COLS if self.AEI else self.DEFAULT_ADI_COLS
        self.srcColNames = getConfigData(self.d_cf, ".X_train_columns", default_cols)
        log.debug("X_train columns: {}".format(self.srcColNames))
        self.srcColNames = [c.strip() for c in self.srcColNames.split(",")]
        if any([col in self.srcColNames for col in self.allNeighborColNames]):
            self.useNeighborFeatures = True
        self.modelColNames = self.srcColNames + [self.tgtColName]

    @staticmethod
    def getRuleModelSetting(cf, prefix='.'):
        rule_model = {}
        rule_model['filters'] = getConfigData(cf, prefix+"/filters", "NeighborParalism<0.98, NeighborOrientation<0.98")
        rule_model['maxTailLength'] = getConfigData(cf, prefix+"/max_tail_length", 20)
        rule_model['smooth'] = getConfigData(cf, prefix+"/smooth", 1) > 0
        return rule_model

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

    def splitDataSet(self):
        # filter 1: costwt 
        df_zerowt = self.d_df_patterns.loc[self.d_df_patterns.costwt<=0, :]
        df = self.d_df_patterns.loc[self.d_df_patterns.costwt>0, :] # valid dataset

        # filter 2: bbox or thresh
        # wi_UserLabel = np.logical_or.reduce( (pd.notnull(df['bbox/Outlier']), pd.notnull(df['bbox/Good']), 
        #                                         pd.notnull(df['threshold/ridge_intensity'])))
        try:
            wi_UserLabel = np.logical_or(pd.notnull(df['bbox/Outlier']), pd.notnull(df['bbox/Good']))
        except KeyError:
            wi_UserLabel = pd.notnull(df['threshold/ridge_intensity'])
        df_wolabel = df.loc[~wi_UserLabel, :]
        df = df.loc[wi_UserLabel, :]
        nsamples = len(df)

        # divide rule
        divide_rule = getConfigData(self.d_cf, ".divide_rule", "60:40:0")
        log.debug("divide_rule: {}".format(divide_rule))
        divide_rule = list(map(float, divide_rule.split(":")))
        divides = ContourSelCalStage.divideSample(nsamples, divide_rule)

        # assign usage
        usages = np.tile(None, (nsamples,))
        usages[:divides[0]] = USAGE_TYPES[0]
        usages[divides[0]:divides[1]] = USAGE_TYPES[1]
        usages[divides[1]:] = USAGE_TYPES[2]
        df = df.assign(usage=usages)
        log.debug("Usages: {} {}".format(len(usages), usages))

        # restore df of patterns
        self.d_df_patterns = pd.concat([df, df_wolabel, df_zerowt], axis=0)

    def loadDataSet(self, modeltype='rule'):
        cal_dataset = []
        ver_dataset = []
        cal_patterns = []
        ver_patterns = []
        cal_file_paths = []
        ver_file_paths = []

        for _, row in self.d_df_patterns.iterrows(): # loop patterns 
            if row.usage in (USAGE_TYPES[:-1]) :
                
                contourfile = self.validateFile(row.loc['contour/path'])
                log.debug('loadDataSet, read unified contour file: {}'.format(contourfile))
                contour = parseContourWrapper(contourfile)
                if contour is None:
                    continue
                curdf = contour.toDf()
                if self.useNeighborFeatures and modeltype != 'rule':
                    curdf = addNeighborFeatures(curdf)

                curdf = curdf.loc[pd.notnull(curdf.loc[:, self.tgtColName]), :] # only use SEM points in ROI
                curdf.loc[:, 'patternid'] = row.loc['name']
                
                # curdf = curdf.loc[:, self.modelColNames]
                if row.loc['usage'] == USAGE_TYPES[0]: # cal pattern SEM points
                    cal_dataset.append(curdf)
                    cal_patterns.append(row.loc['name'])
                    cal_file_paths.append(contourfile)
                elif row.loc['usage'] == USAGE_TYPES[1]: # ver pattern SEM points
                    ver_dataset.append(curdf)
                    ver_patterns.append(row.loc['name'])
                    ver_file_paths.append(contourfile)
        if len(ver_dataset) == 0: # in case of empty ver dataset
            ver_dataset = cal_dataset[-1:]
            ver_patterns = cal_patterns[-1:]
            ver_file_paths = cal_file_paths[-1:]
        assert(len(ver_dataset) != 0 and len(cal_dataset) != 0)

        if modeltype == 'rule':
            return (cal_file_paths, ver_file_paths)

        cal_dataset = pd.concat(cal_dataset)
        ver_dataset = pd.concat(ver_dataset)
        if self.debugOn:
            cal_dataset.loc[:, 'usage'] = 'CAL'
            ver_dataset.loc[:, 'usage'] = 'VER' 
            calset = pd.concat([cal_dataset, ver_dataset])
            caldatapath = os.path.join(self.stageresultabspath, 'caldata.txt')
            calset.to_csv(caldatapath, index=False, sep='\t')
            log.debug("all CAL&VER data are saved at "+caldatapath)

        X_cal, y_cal = cal_dataset[self.srcColNames], cal_dataset[self.tgtColName]
        X_ver, y_ver = ver_dataset[self.srcColNames], ver_dataset[self.tgtColName]
        log.debug("cal set pattern names: {}".format(cal_patterns))
        log.debug("cal set memory info: \n{}".format(ContourSelCalStage.getDfMemoryInfo(cal_dataset)))
        log.debug("ver set pattern names: {}".format(ver_patterns))
        log.debug("ver set memory info: \n{}".format(ContourSelCalStage.getDfMemoryInfo(ver_dataset)))

        return (X_cal, y_cal, X_ver, y_ver)

    @staticmethod
    def getDfMemoryInfo(df):
        buffer = io.StringIO()
        df.info(buf=buffer, verbose=False, memory_usage='deep')
        s = buffer.getvalue()
        return s

    def calibrate(self, modeltype='clf'):
        pickleModelName = "{}_model.pickle".format(modeltype)
        modelClass = None
        if not self.reuseModel:
            if modeltype != 'rule': # Clf model
                X_cal, y_cal, X_ver, y_ver = self.loadDataSet(modeltype='clf')
                log.debug("y_cal values count:\n{}".format(y_cal.value_counts()))
                log.debug("y_ver values count:\n{}".format(y_ver.value_counts()))
                modelClass = ContourSelClfModel()
                model, Xminmax, cm_cal, cm_ver = modelClass.calibrate(X_cal, y_cal, X_ver, y_ver)
                del X_cal, y_cal, X_ver, y_ver
            else: # rule model
                cal_file_paths, ver_file_paths = self.loadDataSet(modeltype='rule')
                model = ContourSelCalStage.getRuleModelSetting(self.d_cf, '.rule_model')
                modelClass = ContourSelRuleModel(**model)
                cm_cal, cm_ver = modelClass.calibrate(cal_file_paths, ver_file_paths)
                Xminmax = None
            ret = (modeltype, model, Xminmax, cm_cal, cm_ver)
            
            # Save model to pickle file
            pkl_filename = os.path.join(self.stageresultabspath, pickleModelName)
            with open(pkl_filename, 'wb') as fout:  
                pickle.dump(ret, fout)

        else: # Load model from pickle file
            pkl_filename = os.path.join(self.stageresultabspath, pickleModelName)
            with open(pkl_filename, 'rb') as fin:  
                ret = pickle.load(fin)
            if ret[0] != 'rule': # Clf model
                modelClass = ContourSelClfModel()
            else:
                modelClass = ContourSelRuleModel()

        modelClass.printModelPerformance(ret[3], usage='CAL')
        modelClass.printModelPerformance(ret[4], usage='VER')
        return ret
    

    def calibrateAssembleModel(self):
        '''
        {stage1: [model1, model2], stage2: [model1, ..]}
        '''
        pass

    def apply(self, modeltype, model, Xminmax=None):
        start = time.time()
        if self.useMultiprocess:
            self.applyMultiprocess(modeltype, model, Xminmax)
        else:
            self.applyTranverse(modeltype, model, Xminmax)
        endtime = time.time()
        log.info('Total apply time(useMultiprocess={}): {}s'.format(self.useMultiprocess, endtime-start))
    
    def applyTranverse(self, modeltype, model, Xminmax=None):
        ## traverse mode model apply, loop each pattern to get the model apply result
        for patternidx, row in self.d_df_patterns.iterrows():
            if row.loc['costwt'] <= 0:
                continue
            contourfile = self.validateFile(row.loc['contour/path'])
            if contourfile is None:
                continue
            self.applySingleContour(modeltype, model, Xminmax, contourfile, patternidx)

    def applySingleContourWraper(self, argv):
        self.applySingleContour(*argv)

    def applySingleContour(self, modeltype, model, Xminmax, contourfile, patternidx):
        patternid = os.path.basename(contourfile).strip('_image_contour.txt')
            
        # apply model into contour
        log.debug("Start processing pattern {} ...".format(patternid))
        contour = parseContourWrapper(contourfile)
        if contour is None:
            return
        curdf = contour.toDf()
        if modeltype != 'rule':
            if self.useNeighborFeatures:
                curdf = addNeighborFeatures(curdf)
            X = curdf[self.srcColNames]
            if Xminmax is not None:
                X = ContourSelClfModel.applyFeatureScalar(X, Xminmax)

            y_pred, cm = ContourSelClfModel.predict(model, X)
            curdf.loc[:, self.outColName] = y_pred
        else: # rule model
            curdf, cm = ContourSelRuleModel.predict(model, curdf)
        curdf = curdf.astype({self.outColName: int})
        FNr, FPr = ContourSelBaseModel.calcFpFnRate(cm)
        gt0flt = curdf.weight > 0
        curdf.loc[gt0flt, 'weight'] = curdf.loc[gt0flt, self.outColName]

        # write new contour
        newcontour = contour.fromDf(curdf)
        newcontourfile = os.path.join(self.stageresultabspath, '{}_image_contour.txt'.format(patternid))
        newcontour.saveContour(newcontourfile)
        newcontourfile_relpath = os.path.join(self.stageresultrelpath, os.path.basename(contourfile))
        self.d_df_patterns.loc[patternidx, 'contour/path'] = newcontourfile_relpath
        self.d_df_patterns.loc[patternidx, 'missing'] = FNr
        self.d_df_patterns.loc[patternidx, 'false_alarm'] = FPr
        log.debug("Successfully processed pattern {}".format(patternid))

    def applyMultiprocess(self, modeltype, model, Xminmax=None):
        argvs = []
        for patternidx, row in self.d_df_patterns.iterrows():
            if row.loc['costwt'] <= 0:
                continue
            contourfile = self.validateFile(row.loc['contour/path'])
            if contourfile is None:
                continue
            argvs.append((modeltype, model, Xminmax, contourfile, patternidx))

        try:
            cpu_used = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_used = 0
        cpu_num = multiprocessing.cpu_count() - cpu_used
        cpu_num = int(0.6 * cpu_num)

        log.info('Using {} processes'.format(cpu_num))
        pool = multiprocessing.pool.ThreadPool(processes=cpu_num)
        pool.imap(self.applySingleContourWraper, argvs)
        pool.close()
        pool.join()

    def applyAssembleModel(self):
        pass

    def run(self):
        # split DataSet into Cal or Ver Usage, and loading data
        self.splitDataSet()

        # model calibration
        modeltype = getConfigData(self.d_cf, '.modeltype', 'rule')
        modeltype, model, Xminmax, _, _ = self.calibrate(modeltype)

        # model apply, calculate all pattern SEM point's info
        if self.applyModel:
            self.apply(modeltype, model, Xminmax)

    def save(self, path, viaDf=True): # override the base save() method, to save via DataFrame
        super(ContourSelCalStage, self).save(path, viaDf=True)



