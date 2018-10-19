# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-25 14:53:41

Contour Selection model calibration stage

Last Modified by:  ouxiaogu
"""

import numpy as np
import pandas as pd
import io
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pickle

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
    srcColNames = 'slope, intensity, ridge_intensity, contrast, EigenRatio'
    tgtColName = 'UserLabel'
    neighborColNames = ['NeighborOrientation', 'NeighborParalism'] # used neigbor filters
    allNeighborColNames = ['NeighborContinuity', 'NeighborOrientation', 'NeighborParalism'] 
    debugOn = True
    pickleName = "pickle_model.pkl"
    reuseModel = True

    def __init__(self, gcf, cf, stagename, jobpath):
        super(ContourSelCalStage, self).__init__(gcf, cf, stagename, jobpath)
        self.__getXTrainCols()

    def __getXTrainCols(self):
        self.srcColNames = getConfigData(self.d_cf, ".X_train_columns", self.srcColNames)
        log.debug("X_train columns: {}".format(self.srcColNames))
        self.srcColNames = [c.strip() for c in self.srcColNames.split(",")]
        if getConfigData(self.d_cf, "use_per_seg_feature", 0) > 0:
            self.usePerSegFeatures = True
        if self.usePerSegFeatures:
            self.srcColNames += self.neighborColNames
        self.modelColNames = self.srcColNames + [self.tgtColName]

    @staticmethod
    def splitSample(nsamples, divide_rule):
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
        df_zerowt = self.d_df.loc[self.d_df.costwt<=0, :]
        df = self.d_df.loc[self.d_df.costwt>0, :] # valid dataset

        # filter 2: bbox 
        wi_bbox = np.logical_or(pd.notnull(df['bbox/Outlier']), pd.notnull(df['bbox/Good']))
        df_wobbox = df.loc[~wi_bbox, :]
        df = df.loc[wi_bbox, :]
        nsamples = len(df)

        # divide rule
        divide_rule = getConfigData(self.d_cf, ".divide_rule", "60:40:0")
        log.debug("divide_rule: {}".format(divide_rule))
        divide_rule = list(map(float, divide_rule.split(":")))
        divides = ContourSelCalStage.splitSample(nsamples, divide_rule)

        # assign usage
        usages = np.tile(None, (nsamples,))
        usages[:divides[0]] = USAGE_TYPES[0]
        usages[divides[0]:divides[1]] = USAGE_TYPES[1]
        usages[divides[1]:] = USAGE_TYPES[2]
        df = df.assign(usage=usages)
        log.debug("Usages: {} {}".format(len(usages), usages))

        # restore df
        self.d_df = pd.concat([df, df_wobbox, df_zerowt], axis=0)

    def loadDataSet(self):
        cal_dataset = []
        ver_dataset = []
        cal_patterns = []
        ver_patterns = []

        for _, row in self.d_df.iterrows(): # loop pattern ocf
            if row.usage in (USAGE_TYPES[:-1]) :
                contour = SEMContour()
                contourfile = os.path.join(self.jobresultabspath, row.loc['contour/path'])
                if contour.parseFile(contourfile):
                    curdf = contour.toDf()
                    if self.usePerSegFeatures:
                        curdf = self.addNeighborFeatures(curdf)
                    curdf = curdf.loc[pd.notnull(curdf.loc[:, self.tgtColName]), :] # only use SEM points in ROI
                    if len(curdf) == 0:
                        continue
                    # curdf = curdf.loc[:, self.modelColNames]
                    if row.loc['usage'] == USAGE_TYPES[0]: # cal pattern SEM points
                        cal_dataset.append(curdf)
                        cal_patterns.append(row.loc['name'])
                    elif row.loc['usage'] == USAGE_TYPES[1]: # ver pattern SEM points
                        ver_dataset.append(curdf)
                        ver_patterns.append(row.loc['name'])

        cal_dataset = pd.concat(cal_dataset)
        if self.debugOn:
            caldatapath = os.path.join(self.stageresultabspath, 'caldata.txt')
            cal_dataset.to_csv(caldatapath, index=False, sep='\t')
            log.debug("training data set saved at "+caldatapath)
        ver_dataset = pd.concat(ver_dataset)

        X_cal, y_cal = cal_dataset[self.srcColNames], cal_dataset[self.tgtColName]
        X_ver, y_ver = ver_dataset[self.srcColNames], ver_dataset[self.tgtColName]
        log.debug("cal set pattern names: {}".format(cal_patterns))
        log.debug("cal set memory info: \n{}".format(ContourSelCalStage.getDfMemoryInfo(cal_dataset)))
        log.debug("ver set pattern names: {}".format(ver_patterns))
        log.debug("ver set memory info: \n{}".format(ContourSelCalStage.getDfMemoryInfo(ver_dataset)))

        return X_cal, y_cal, X_ver, y_ver

    @staticmethod
    def getDfMemoryInfo(df):
        buffer = io.StringIO()
        df.info(buf=buffer, verbose=False, memory_usage='deep')
        s = buffer.getvalue()
        return s

    @staticmethod
    def calcRMS(y_pred, y):
        return np.sqrt(np.mean(np.power(y_pred - y, 2)))

    def addNeighborFeatures(self, df):
        '''
        add Features for the input contour DataFrame, based on the neighbor relationship in the context of segment

        Parameters:
        -----------
        df: [in, out] contour as DataFrame
            [in] Contour df, must contains `polygonId`, `angle`, `offsetx`, `offsety`
            [out] Contour df, added `NeighborContinuity`, `NeighborOrientation`, `NeighborParalism`

                - `NeighborContinuity`:  |X(n) - X(n-1)|^2, usually is to 1 (because of 8-neighbor contour tracing)
                - `NeighborOrientation`:  dot(EigenVector(n), EigenVector(n-1)), closer to 1, the better(may use 1-dot)
                - `NeighborParalism`:  ||cross((X(n) - X(n-1)), EigenVector(n-1))||, closer to 1, the better(may use 1-cross)
        TODO, the segment neighborhood based features can only be obtained by the whole segment, can't use ROI cropped segment 
        '''
        polygonIds = df.loc[:, 'polygonId'].drop_duplicates().values
        preIdx = df.index[0]
        for polygonId in polygonIds:
            isPolygonHead = True
            for curIdx, _ in df.loc[df['polygonId']==polygonId, :].iterrows():
                NeighborContinuity = 0
                NeighborOrientation = 1
                NeighborParalism = 1
                if not isPolygonHead:
                    eigenvector_n_1 = np.array([np.cos(df.loc[preIdx, 'angle']), np.sin(df.loc[preIdx, 'angle'])])
                    eigenvector_n = np.array([np.cos(df.loc[curIdx, 'angle']), np.sin(df.loc[curIdx, 'angle'])])
                    neighorvector = np.array([df.loc[curIdx, 'offsetx'] - df.loc[preIdx, 'offsetx'],
                                            df.loc[curIdx, 'offsety'] - df.loc[preIdx, 'offsety']])
                    crossvector = np.cross(neighorvector, eigenvector_n_1)

                    NeighborContinuity = np.sqrt(neighorvector.dot(neighorvector))
                    NeighborOrientation = eigenvector_n.dot(eigenvector_n_1)
                    NeighborParalism = np.sqrt(crossvector.dot(crossvector))/NeighborContinuity
                    NeighborContinuity = abs(1-NeighborContinuity)
                preIdx = curIdx
                isPolygonHead = False
                
                for ii, val in enumerate([NeighborContinuity, NeighborOrientation, NeighborParalism]):
                    colname = self.allNeighborColNames[ii]
                    if colname in self.neighborColNames:
                        df.loc[curIdx, colname] = val
        return df

    def calibrate(self):
        # split DataSet into Cal or Ver Usage, and loading data
        self.splitDataSet()
        X_cal, y_cal, X_ver, y_ver = self.loadDataSet()
        log.debug("y_cal values count:\n{}".format(y_cal.value_counts()))
        log.debug("y_ver values count:\n{}".format(y_ver.value_counts()))
        
        # classification model
        clf = svm.SVC(kernel='linear', class_weight='balanced') # {0: 10, 1: 1}
        model = clf.fit(X_cal, y_cal)
        log.info("SVC model, coef: {}, intercept: {}".format(clf.coef_, clf.intercept_))

        # cal data
        y_cal_pred = model.predict(X_cal)
        rms = ContourSelCalStage.calcRMS(y_cal_pred, y_cal)
        cm = confusion_matrix(y_cal, y_cal_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        log.info("SVC model rms on calibration set: {}".format(rms))
        log.info("SVC model confusion matrix on calibration set:\n{}\n{}".format(cm, cm_norm))

        # ver data
        y_ver_pred = model.predict(X_ver)
        rms = ContourSelCalStage.calcRMS(y_ver_pred, y_ver)
        cm = confusion_matrix(y_ver, y_ver_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        log.info("SVC model rms on verification set: {}".format(rms))
        log.info("SVC model confusion matrix on verification set:\n{}\n{}".format(cm, cm_norm))
        del X_cal, y_cal, X_ver, y_ver

        # Save model to pickle file
        pkl_filename = os.path.join(self.stageresultabspath, self.pickleName)
        with open(pkl_filename, 'wb') as file:  
            pickle.dump(model, file)
        return model

    def predict(self, model):

        for idx, row in self.d_df.iterrows():
            contour = SEMContour()
            contourfile = os.path.join(self.jobresultabspath, row.loc['contour/path'])
            if contour.parseFile(contourfile):
                curdf = contour.toDf()
                if self.usePerSegFeatures:
                    curdf = self.addNeighborFeatures(curdf)
                X = curdf[self.srcColNames]
                y_pred = model.predict(X)
                curdf.loc[:, 'ClfLabel'] = y_pred
                if self.tgtColName in curdf.columns:
                    y_true = curdf[self.tgtColName]
                    rms = ContourSelCalStage.calcRMS(y_pred, y)
                    self.d_df.loc[idx, 'clf_rms'] = rms

                newcontour = contour.fromDf(curdf)
                patternid = self.d_df.loc[idx, 'name']
                newcontourfile_relpath = os.path.join(self.stageresultrelpath, '{}_image_contour.txt'.format(patternid))
                newcontourfile = os.path.join(self.jobresultabspath, newcontourfile_relpath)
                try:
                    newcontour.saveContour(newcontourfile)
                except FileNotFoundError:
                    print("jobresultabspath: ", self.jobresultabspath)
                    print("old contour file: ", contourfile)
                    print("new contour file: ", newcontourfile)
                    raise

                self.d_df.loc[idx, 'contour/path'] = newcontourfile_relpath

    def run(self):
        # model calibration
        if not self.reuseModel:
            model = self.calibrate()
        else:
            # Load model from pickle file
            pkl_filename = os.path.join(self.stageresultabspath, self.pickleName)
            with open(pkl_filename, 'rb') as file:  
                model = pickle.load(file)

        # model predict, calculate all pattern SEM point's info
        self.predict(model)

    def save(self, path, viaDf=True): # override the base save() method, to save via DataFrame
        super(ContourSelCalStage, self).save(path, viaDf=True)