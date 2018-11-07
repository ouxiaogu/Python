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
from logger import logger
log = logger.getLogger(__name__)

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

    @staticmethod
    def calcFpFnRate(cm):
        '''calculate RMS by confusion matrix'''
        FN, FP = cm[0, 1], cm[1, 0]
        UserPositive, UserNegative = cm.sum(axis=1).astype(float)
        FP = FP / UserNegative if UserNegative != 0 else np.nan
        FN = FN / UserPositive if UserPositive != 0 else np.nan
        return FP, FN

    @classmethod
    def printModelPerformance(cls, cm, usage='CAL'):
        cm_norm = cm.astype('float') / cm.sum(axis=1).reshape((2, 1))
        log.info("{} model on {} set, FN(missing) rate = {:.3f}%, FP(false alarm) rate = {:.3f}%".format(cls.modeltype, usage, cm_norm[0, 1]*100, cm_norm[1, 0]*100))
        log.info("{} model confusion matrix on {} set:\n{}\n{}".format(cls.modeltype, usage, cm, 100*cm_norm))

    @staticmethod
    def computeConfusionMatrix(contourdf):
        ''' compute the confusion matrix in below format
            [TP      FN]
            [FP      TN]
        '''
        tgtColName, outColName = ContourSelBaseModel.tgtColName, ContourSelBaseModel.outColName
        cm = np.zeros((2,2), dtype=int)
        if (not contourdf.empty) and all([col in contourdf.columns for col in [tgtColName, outColName]]):
            TP = (contourdf.loc[:, tgtColName] == 0) & (contourdf.loc[:, outColName] == 0)
            FN = (contourdf.loc[:, tgtColName] == 0) & (contourdf.loc[:, outColName] == 1)
            FP = (contourdf.loc[:, tgtColName] == 1) & (contourdf.loc[:, outColName] == 0)
            TN = (contourdf.loc[:, tgtColName] == 1) & (contourdf.loc[:, outColName] == 1)

            cm[0, 0] = len(contourdf.loc[TP, :])
            cm[0, 1] = len(contourdf.loc[FN, :])
            cm[1, 0] = len(contourdf.loc[FP, :])
            cm[1, 1] = len(contourdf.loc[TN, :])
        return cm