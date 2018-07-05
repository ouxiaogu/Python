'''
-*- coding: utf-8 -*-
Created: peyang, 2018-01-15 16:07:20

Last Modified by: ouxiaogu

ImDescriptors: Image Descriptors Module

Part 1: some histogram functions:
This is a sample for histogram plotting for RGB images and grayscale images for better understanding of colour distribution

Benefit : Learn how to draw histogram of images
          Get familiar with cv2.calcHist, cv2.equalizeHist,cv2.normalize and some drawing functions

Level : Beginner or Intermediate

Functions : 1) hist_curve : returns histogram of an image drawn as curves
            2) hist_lines : return histogram of an image drawn as bins ( only for grayscale images )

Usage : python hist.py <image_file>

Abid Rahman 3/14/12 debug Gary Bradski

Part 2: Image merge overview functions
'''

from sys import platform
import numpy as np
from FileUtil import FileScanner, getFileLabel
from subprocess import call
import logger
import os.path
import cv2
import pandas as pd
import re

__all__ = [ 'RMS_BIN_RANGES', 'ZNCC_BIN_RANGES',
            'calcHist', 'readDumpImage', 'readBBox',
            ]

RMS_BIN_RANGES = [0, 2, 4, 6, 8, 10, 15, 20, 30, 50, 100]
ZNCC_BIN_RANGES = [0, 0.2, 0.5, 0.8]

logger.initlogging(debug=False)
log = logger.getLogger("ImDescriptors")
BINS = np.arange(256).reshape(256,1)

def hist_curve(im):
    '''return histogram of an image drawn as curves'''
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((BINS,hist)))
        cv2.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(im):
    '''return histogram of an image drawn as BINS ( only for grayscale images )
    hist_item is normalized into [0, 255]
    the hist height is set as 300 as default, a little higher than max->255
    '''
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y

def hist_rect(im, hbins=100):
    '''return histogram of any image as some rectangle bins'''
    im = im.astype('float32')
    histSize = [hbins]
    minVal = np.amin(im)
    maxVal = np.amax(im)
    histRange = np.asarray([minVal, maxVal])
    hist_item = cv2.calcHist([im], [0], None, histSize, histRange.astype('float32'))
    cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))

    scale = 301/hbins
    histImg = np.zeros((300, hbins*scale, 3))
    for ix in range(hbins):
        binVal = hist[ix]
        cv2.rectangle(histImg, (ix*scale, 0), ((ix+1)*scale, binVal), (255,255,255))
    histImg = np.flipud(histImg)
    return histImg

def calcHist(series_, ranges=RMS_BIN_RANGES, column=None):
    """
    calculate the histogram of given Pandas Serires

    Parameters
    ----------
    series_ : Pandas Series to be analyzed
    ranges : Array of the histogram bin boundaries.

    Returns
    -------
    histDF : DataFrame of the hists
    """

    labels = []
    filts = []
    hists = np.zeros((len(series_), len(ranges)))
    nitems = len(ranges)
    log.debug("ranges, %s" % str(ranges))

    for i in range(nitems):
        if i == nitems - 1:
            label = "{}~".format(ranges[i])
            filt  = lambda x, y=ranges[i]: (x >= y) & True
        else:
            label = "{}~{}".format(ranges[i], ranges[i+1])
            filt  = lambda x, y=ranges[i], z=ranges[i+1]: (x >= y) & (x < z)
        labels.append(label)
        filts.append(filt)
    log.debug(labels)
    log.debug("%s" % str([ff==filts[0] for ff in filts]))

    for i, series in enumerate(series_):
        if column is not None:
            try:
                series = series.ix[:, column]
            except KeyError:
                raise KeyError("Input column {} is not in columns {}".format(series.columns))
        if not isinstance(series, pd.Series):
            try:
                series = pd.Series(series)
            except ValueError:
                raise ValueError("Error occurs when converting input into Pandas Series")
        series = series.sort_values()
        log.debug("series_: %s" % str(series.values ))
        for j, filt in enumerate(filts):
            hists[i, j] = len(series.ix[ series.apply(filt) ] )
            log.debug("%s: %d" % (labels[j], hists[i, j]))
    histDF = pd.DataFrame(hists, columns=labels).astype('int32')
    histDF = histDF.transpose() # for barplot, column as range lables
    return histDF


