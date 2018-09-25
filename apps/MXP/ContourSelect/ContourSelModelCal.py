# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-25 14:53:41

Contour Selection model calibration stage

Last Modified by:  ouxiaogu
"""

import numpy as np
import pandas as pd
from sklearn import svm

import sys
import os.path
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/tacx/")
from SEMContour import SEMContour, ContourBBox
from MxpStage import MxpStage

sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/common/")
from XmlUtil import addChildNode, getConfigData, setConfigData
import logger
log = logger.setup("ContourModelCal", 'debug')

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

    TODO, if without this Contour Labeling Stage, use the MXP_Flag as UserLabel,
    MXP_Flag with filter BitMask to label 'good' & 'bad'.
    """
    srcColNames = 'intensity  ridge_intensity   contrast    EigenRatio'.split()
    tgtColName = 'UserLabel'

    def splitDataSet(self):
        df = self.d_df.loc[self.d_df.costwt>0, :] # valid dataset
        df_skipped = self.d_df.loc[self.d_df.costwt<=0, :]
        nsamples = len(df)

        # divide rule
        divide_rule = getConfigData(self.d_cf, ".divide_rule", "60:40:0")
        divide_rule = list(map(int, divide_rule.split(":")))
        if sum(divide_rule) != 100:
            log.warning("Warning, sum of training set, validation set and test set should be 100, input is %s, use 60:40:0 instead." % ':'.join(divide_rule))
        divides = [max(1, int(d/100.*nsamples)) for d in divide_rule]
        divides[-1] = nsamples - sum(divides[:-1])
        assert(all(np.array(divides)>=0))
        divides = np.cumsum(divides)

        # assign usage
        usages = np.tile(None, (nsamples,))
        usages[:divides[0]] = USAGE_TYPES[0]
        usages[divides[0]:divides[1]] = USAGE_TYPES[1]
        usages[divides[1]:] = USAGE_TYPES[2]
        df = df.assign(usage=usages)
        log.debug("Usages: {} {}".format(len(usages), usages))

        # restore df
        self.d_df = pd.concat([df, df_skipped], axis=0)

    def loadDataSet(self):
        cal_df = self.d_df.loc[self.d_df.usage==USAGE_TYPES[0], :]
        log.debug("cal set pattern names: {}".format(cal_df['name'].values))
        cal_dataset = []
        for _, row in cal_df.iterrows():
            contour = SEMContour()
            contourfile = os.path.join(self.jobresultabspath, row.loc['contour/path'])
            if contour.parseFile(contourfile):
                cal_dataset.append(contour.toDf())
        cal_dataset = pd.concat(cal_dataset)
        X_cal, y_cal = cal_dataset[self.srcColNames], cal_dataset[self.tgtColName]

        ver_df = self.d_df.loc[self.d_df.usage==USAGE_TYPES[1], :]
        log.debug("ver set pattern names: {}".format(ver_df['name'].values))
        ver_dataset = []
        for _, row in ver_df.iterrows():
            contour = SEMContour()
            contourfile = os.path.join(self.jobresultabspath, row.loc['contour/path'])
            if contour.parseFile(contourfile):
                ver_dataset.append(contour.toDf())
        ver_dataset = pd.concat(ver_dataset)
        X_ver, y_ver = ver_dataset[self.srcColNames], ver_dataset[self.tgtColName]
        return X_cal, y_cal, X_ver, y_ver

    def run(self):
        from sklearn.metrics import confusion_matrix
        self.splitDataSet()

        X_cal, y_cal, X_ver, y_ver = self.loadDataSet()
        
        clf = svm.SVC(kernel='linear', class_weight={0: 10})
        model = clf.fit(X_cal, y_cal)
        log.info("SVC model, coef: {}, intercept: {}".format(clf.coef_, clf.intercept_))
        rmsfunc = lambda y, y0 : np.sqrt(1 / len(y0) * np.sum(np.power(y - y0, 2)))

        # cal data
        y_cal_pred = model.predict(X_cal)
        rms = rmsfunc(y_cal_pred, y_cal)
        cnf_matrix = confusion_matrix(y_cal, y_cal_pred)
        cm_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        log.info("SVC model rms on calibration set: {}".format(rms))
        log.info("SVC model confusion matrix on calibration set:\n{}\n{}".format(cnf_matrix, cm_norm))

        # cal data
        y_ver_pred = model.predict(X_ver)
        rms = rmsfunc(y_ver_pred, y_ver)
        cnf_matrix = confusion_matrix(y_ver, y_ver_pred)
        cm_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        log.info("SVC model rms on verification set: {}".format(rms))
        log.info("SVC model confusion matrix on verification set:\n{}\n{}".format(cnf_matrix, cm_norm))
        del X_cal, y_cal, X_ver, y_ver

        # per pattern info
        for idx, row in self.d_df.iterrows():
            contour = SEMContour()
            contourfile = os.path.join(self.jobresultabspath, row.loc['contour/path'])
            if contour.parseFile(contourfile):
                curdf = contour.toDf()
                X, y = curdf[self.srcColNames], curdf[self.tgtColName]
                y_pred = model.predict(X)
                rms = rmsfunc(y_pred, y)
                curdf.loc[:, 'clf_label'] = y_pred
                self.d_df.loc[idx, 'clf_rms'] = rms
                newcontour = contour.fromDf(curdf)
                patternid = self.d_df.loc[idx, 'name']
                newcontourfile_relpath = os.path.join(self.stageresultrelpath, '{}_image_contour.txt'.format(patternid))
                newcontourfile = os.path.join(self.jobresultabspath, newcontourfile_relpath)
                newcontour.saveContour(newcontourfile)
                self.d_df.loc[idx, 'contour/path'] = newcontourfile_relpath

    def save(self, path, viaDf=True):
        super(ContourSelCalStage, self).save(path, viaDf=True)