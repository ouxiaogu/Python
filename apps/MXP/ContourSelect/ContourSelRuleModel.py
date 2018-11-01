# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-10-31 17:09:04

The class for contour selection rule model
The rule is built based on the MXP product, good to use MXP classes like SEMContour by design

Last Modified by:  ouxiaogu
"""

import numpy as np
import pandas as pd

from ContourSelBaseModel import ContourSelBaseModel

import sys
import os.path
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/tacx/")
from SEMContour import SEMContour
sys.path.insert(0, (os.path.dirname(os.path.abspath(__file__)))+"/../../../libs/common/")
import logger
log = logger.setup("ContourSelRuleModel", 'debug')

tgtColName = 'UserLabel'
outColName = 'ClfLabel'
neighborColNames = ['NeighborOrientation', 'NeighborParalism'] # used neighbor filters
allNeighborColNames = ['NeighborContinuity', 'NeighborOrientation', 'NeighborParalism'] 

def addNeighborFeatures(df):
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
    if len(df) <= 0:
        return df
    polygonIds = df.loc[:, 'polygonId'].drop_duplicates().values
    preIdx = df.index[0]
    for polygonId in polygonIds:
        isPolygonHead = True
        for curIdx, _ in df.loc[df['polygonId']==polygonId, :].iterrows():
            NeighborContinuity = 1
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
                NeighborContinuity = NeighborContinuity
            preIdx = curIdx
            isPolygonHead = False
            for ii, val in enumerate([NeighborContinuity, NeighborOrientation, NeighborParalism]):
                colname = allNeighborColNames[ii]
                df.loc[curIdx, colname] = val
    return df

def categorizeFilters(filters):
    if filters is None:
        newfilters = {}
    elif not isinstance(filters, dict):
        newfilters = {}
        for col in allNeighborColNames:
            for strFlt in filters:
                if col in strFlt:
                    newfilters[col] = strFlt
                    break
    filters = newfilters
    # print(filters)
    return filters

def cv_gaussian_kernel(ksize, sigma=0, dtype=None):
    try:
        ksize = int(ksize)
    except:
        raise TypeError("cv_gaussian_kernel is to generate linear Gaussian filter, ksize should be int, but input is: {}!\n".format(str(ksize)))
    if ksize <= SMALL_GAUSSIAN_SIZE and sigma <= 0:
        dst = np.array(SMALL_GAUSSIAN_TAB[ksize>>1])
    else:
        if ksize % 2 == 0:
            tmp = ksize + 1
            sys.stderr.write("Warning, kernel size should be odd, adjust ksize from {} to {}!\n".format(ksize, tmp))
            ksize = tmp
        fltSz = ksize
        ksize = float(ksize)
        if(sigma <= 0):
            sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
        func_G = lambda i: math.exp(- (i - (ksize-1)/2)**2 / (2*sigma**2) )
        flt_G = np.asarray(list(func_G(i) for i in range(fltSz)) )
        a = np.sum(flt_G)
        dst = flt_G/a

    if dtype is not None:
        if 'int' in str(dtype):
            vmin = np.min(dst)
            dst = np.floor(dst/vmin + 0.5)
        dst = dst.astype(dtype)
    return dst

def smoothSignal(arr, sigma=0.5):
    # Gaussian kernel
    flt = cv_gaussian_kernel(3, 0.5)

    # padding replicate mode
    padArr = np.zeros((arr.shape[0]+2,) )
    padArr[0] = arr[0]
    padArr[1:-1] = arr
    padArr[-1] = arr[-1]
    
    # smooth
    newArr = np.convolve(padArr, flt, 'valid')
    if len(arr) != len(newArr):
        raise ValueError("unequal length {} {}".format(len(arr), len(newArr)))
    return newArr

def calcMeanOfLargestHistBin(arr, bins=10):
    hist, bin_edges = np.histogram(arr, bins=bins)
    idxmax = np.argmax(hist)
    binvals = arr[np.where(np.logical_and(arr>=bin_edges[idxmax], arr<bin_edges[idxmax+1]))]
    return np.mean(binvals)

def findIndexOfFirstZeroCrossing(arr, gradients=None, start_pos=0):
    if gradients is None:
        gradients = np.gradient(arr, edge_order=2)
    assert(len(arr) == len(gradients))
    start = start_pos+1 if start_pos == 0 else start_pos
    for ix in range(start_pos, len(gradients)):
        if gradients[ix]*gradients[ix-1] <=0 :
            return ix
    return start_pos

def findIndexOfFirstFlat(arr, gradients=None, start_pos=0, thres=None):
    if gradients is None:
        gradients = np.gradient(arr, edge_order=2)
    assert(len(arr) == len(gradients))
    absGradients = np.abs(gradients)
    if thres is None:
        thres = calcMeanOfLargestHistBin(absGradients)
    for ix in range(start_pos, len(gradients)):
        if absGradients[ix] < thres:
            return ix
    return start_pos

def applyNeighborRuleModelPerVLine(linedf, filters='', maxTailLenth=20, smooth=True):
    dominant_issues = []
    linedf.loc[:, outColName] = 1

    # step 1, search and apply from head
    headdf = linedf.loc[linedf.index[:maxTailLenth], :]
    minNeighborOrientation, minNeighborParalism = headdf.min()[allNeighborColNames[1:]]
    issue_feature, issue_index = None, None
    if minNeighborParalism < minNeighborOrientation and len(headdf.query(filters['NeighborParalism'])) > 0:
        issue_feature = 'NeighborParalism'
        issue_index = np.argmin(headdf[issue_feature].values)
    elif minNeighborOrientation < minNeighborParalism and len(headdf.query(filters['NeighborOrientation'])) > 0:
        issue_feature = 'NeighborOrientation'
        issue_index = np.argmin(headdf[issue_feature].values)
    dominant_issues.append(None)
    if issue_feature is not None:
        dominant_issues[0] = [issue_feature, issue_index]
        arr = linedf[issue_feature].values
        if smooth:
            arr = smoothSignal(arr)
        gradient = np.gradient(arr, edge_order=2)
        idxFlat = findIndexOfFirstFlat(arr, gradient, start_pos=issue_index+1)
        dominant_issues[0].append(idxFlat)
        idxFlat = min(maxTailLenth, idxFlat)
        linedf.loc[linedf.index[:idxFlat], outColName] = 0

    # step 2, search and apply from tail, reverse order
    dominant_issues.append(None)
    head_index = 0 if issue_index is None else issue_index
    tailrange = len(linedf) - (head_index + 1) # exclude the head issue index itself
    if tailrange > maxTailLenth:
        taildf = linedf.loc[linedf.index[-maxTailLenth:], :]
        minNeighborOrientation, minNeighborParalism = taildf.min()[allNeighborColNames[1:]]
        issue_feature, issue_index = None, None
        if minNeighborParalism < minNeighborOrientation and len(taildf.query(filters['NeighborParalism'])) > 0:
            issue_feature = 'NeighborParalism'
            issue_index = np.argmin(taildf[issue_feature].values)
            issue_index = maxTailLenth - 1 - issue_index  # use index start from tail
        elif minNeighborOrientation < minNeighborParalism and len(taildf.query(filters['NeighborOrientation'])) > 0:
            issue_feature = 'NeighborOrientation'
            issue_index = np.argmin(taildf[issue_feature].values)
            issue_index = maxTailLenth - 1 - issue_index
        if issue_feature is not None and issue_index:
            dominant_issues[1] = [issue_feature, issue_index]
            arr = linedf[issue_feature].values[::-1]
            if smooth:
                arr = smoothSignal(arr)
            gradient = np.gradient(arr, edge_order=2)
            idxFlat = findIndexOfFirstFlat(arr, gradient, start_pos=issue_index+1)
            dominant_issues[1].append(idxFlat)
            idxFlat = min(maxTailLenth, idxFlat)
            linedf.loc[linedf.index[-idxFlat:], outColName] = 0
    return linedf, dominant_issues

def applyNeighborRuleModelPerVContour(contourdf, filters='', maxTailLenth=20, smooth=True):
    '''
    The step to find rule model in python:
    1. apply combined filters to find ill contour Vline candidates
    2. find the Vline candidates have dominant issues in its head+20 or tail-20
    3. remove contour issue head/tail by following rules(default is 3.1):
        * 3.1: search start from dominant issue position, new head=Index[1st flat gradient point]
        * 3.2: Index[dominant issue position], new head=Index[the gradient zero-crossing point]
    '''
    contourdf.loc[:, outColName] = 1
    filters = categorizeFilters(filters)

    inflection_df = contourdf.query('or '.join(filters.values()))
    # print(inflection_df[['polygonId', 'offsetx', 'offsety'] + allNeighborColNames[1:]])
    polygonIds = inflection_df.loc[:, 'polygonId'].drop_duplicates().values
    for polygonId in polygonIds:
        lineFlt = contourdf['polygonId']==polygonId
        linedf = contourdf.loc[lineFlt, :]
        newlinedf, dominant_issues = applyNeighborRuleModelPerVLine(linedf, filters, maxTailLenth, smooth)
        # print(int(polygonId), dominant_issues)
        contourdf.loc[lineFlt, :] = newlinedf
    return contourdf

class ContourSelRuleModel(object):
    '''
    classification model type here includes {'SVC': 'SVM', 'DT': 'Decision Tree', 'RF': 'Random Forest'}
    '''

    def __init__(self, **kwargs):
        self.filters = kwargs.get('filters', "NeighborParalism<0.98, 'NeighborOrientation<0.98")
        self.maxTailLenth = kwargs.get('maxTailLenth', 20)
        self.smooth = kwargs.get('smooth', 1)

    @staticmethod
    def getModelType():
        return "rule"

    def getRuleModel(self):
        rule_model = {'filters': self.filters, 'maxTailLenth': self.maxTailLenth, 'smooth': self.smooth}
        return rule_model

    def calibrate(self, cal_file_paths, ver_file_paths):
        '''
        traverse model, loop files to get the calibration performance
        '''

        # get rule model from user input/default value
        model = self.getRuleModel()

        # calibration performance
        cm_cal = ContourSelRuleModel.checkModel(model, cal_file_paths, usage='CAL')
        cm_ver = ContourSelRuleModel.checkModel(model, ver_file_paths, usage='VER')
        self.printModelPerformance(cm_cal, usage='CAL')
        self.printModelPerformance(cm_ver, usage='VER')

        return cm_cal, cm_ver

    @staticmethod
    def checkModel(model, contourfiles, usage='CAL'):
        cm_final = np.zeros((2,2), dtype=int)
        for contourfile in cal_file_paths:
            contour = SEMContour()
            contour.parseFile(contourfile)
            contourdf = contour.toDf()
            _, cm = ContourSelRuleModel.predict(model, contourdf)
            cm_final += cm
        return cm_final

    @staticmethod
    def predict(rule_model, contourdf):
        contourdf = applyNeighborRuleModelPerVContour(contourdf, **rule_model)
        cm = ContourSelRuleModel.computeConfusionMatrix(contourdf)
        return contourdf, cm

    @staticmethod
    def computeConfusionMatrix(contourdf):
        ''' compute the confusion matrix in below format
            [TP      FN]
            [FP      TN]
        '''
        cm = np.zeros((2,2), dtype=int)
        if not (contourdf.empty or any([col not in contourdf.columns for col in [tgtColName, outColName]])):
            TP = (contourdf.loc[:, tgtColName] == 0) & (contourdf.loc[:, outColName] == 0)
            FN = (contourdf.loc[:, tgtColName] == 0) & (contourdf.loc[:, outColName] == 1)
            FP = (contourdf.loc[:, tgtColName] == 1) & (contourdf.loc[:, outColName] == 0)
            TN = (contourdf.loc[:, tgtColName] == 1) & (contourdf.loc[:, outColName] == 1)

            cm[0, 0] = len(contourdf.loc[TP, :])
            cm[0, 1] = len(contourdf.loc[FN, :])
            cm[1, 0] = len(contourdf.loc[FP, :])
            cm[1, 1] = len(contourdf.loc[TN, :])
        return cm