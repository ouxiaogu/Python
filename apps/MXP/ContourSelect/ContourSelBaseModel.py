# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-10-31 11:11:40

The class for contour selection classification model

Last Modified by:  ouxiaogu
"""
import numpy as np

import sys
import os.path

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/common/")
import logger
log = logger.setup("ContourSelBaseModel", 'debug')

class ContourSelBaseModel(object):
    '''
    The base model class for sklearn Classifier and neighbor feature rule model
    but the inheritance relation is weak
    '''
    allSrcColNames = 'slope, intensity, ridge_intensity, contrast, EigenRatio'
    srcColNames = 'slope, intensity, ridge_intensity, contrast'
    tgtColName = 'UserLabel'
    outColName = 'ClfLabel'
    neighborColNames = ['NeighborOrientation', 'NeighborParalism'] # used neighbor filters
    allNeighborColNames = ['NeighborContinuity', 'NeighborOrientation', 'NeighborParalism'] 
    debugOn = True

    def __init__(self, *argv, **kwargs):
        pass

    @staticmethod
    def calcRMS(cm):
        '''calculate RMS by confusion matrix'''
        FN, FP = cm[0, 1], cm[1, 0]
        total = cm.sum().astype(float)
        rms = np.sqrt((FN + FP) / total ) if total != 0 else np.nan

        return rms

    @classmethod
    def printModelPerformance(cls, cm, usage='CAL'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        rms = ContourSelBaseModel.calcRMS(cm)
        log.info("{} model rms on {} set: {}".format(cls.modeltype, usage, rms))
        log.info("{} model confusion matrix on {} set:\n{}\n{}".format(cls.modeltype, usage, cm, cm_norm))