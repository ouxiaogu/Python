# -*- coding: utf-8 -*-
'''
Created: peyang, 2018-01-15 16:07:20

ImDescriptors: Image Descriptors Module

Last Modified by: ouxiaogu
'''

import numpy as np
import cv2
import pandas as pd

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../common")

import logger
log = logger.setup(level='info')

__all__ = [ 'RMS_BIN_RANGES', 'ZNCC_BIN_RANGES', 'calcHistSeries',
            'im_fft_amplitude_phase', 'power_ratio_in_cutoff_frequency',
            'printImageInfo'
        ]

BINS = np.arange(256).reshape(256,1)
RMS_BIN_RANGES = [0, 2, 4, 6, 8, 10, 15, 20, 30, 50, 100]
ZNCC_BIN_RANGES = [0, 0.2, 0.5, 0.8]

def printImageInfo(im):
    shape = 'X'.join(map(str, im.shape))
    print(shape, im.dtype, np.percentile(im, np.linspace(0, 100, 6),  interpolation='nearest'), sep=', ')

def hist_curve(im):
    '''return histogram of an image drawn as curves'''
    h = np.zeros((256,256,3))
    if len(im.shape) == 2:
        h = np.zeros((256,256))
        colors = [(255,255)]
    elif im.shape[2] == 3:
        colors = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, colr in enumerate(colors):
        hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((BINS,hist)))
        cv2.polylines(h,[pts],False, colr)
    h=np.flipud(h)
    return h

def hist_lines(im):
    '''return histogram of an image drawn as BINS ( only for grayscale images )
    hist_item is normalized into [0, 255]
    the hist height is set as 300 as default, a little higher than max->255
    '''

    # h = np.zeros((300,256,3), dtype=np.uint8)
    h = np.zeros((256,256), dtype=np.uint8) # output histogram in grayscale
    if len(im.shape)!=2:
        sys.stderr.write("hist_lines applicable only for grayscale images!\n")
        #print("so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)

    hist=np.uint8(np.around(hist_item)) # normalized hist as CV_U8
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255)) # (255,255,255)
    h = np.flipud(h)
    return h

def hist_rect(im=None, hbins=100, hist=None):
    '''return histogram of any image as some rectangle bins
    In python, it seems cv2 don't support the float type image,
    But for c++, it's supported

    Parameters
    ----------
    hist : None or 1D list-like
        if hist is None, use OpenCV to calcHist
        if hist is 1D list, it should be grayscale histogram, list or dict
    '''
    if hist is None:
        if len(im.shape)!=2:
            sys.stderr.write("hist_lines applicable only for grayscale images!\n")
            #print("so converting image to grayscale for representation"
            im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        hist_item = cv2.calcHist([im], [0], None, [hbins], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
    else:
        hist_item = hist
        hbins = len(hist_item)
    hist = np.uint8(np.around(hist_item)) # normalized hist as CV_U8

    assert(hbins <= 256)
    scale = 256//hbins
    histImg = np.zeros((256, hbins*scale), np.uint8)
    for ix in range(hbins):
        binVal = hist[ix]
        cv2.rectangle(histImg, (ix*scale, 0), ((ix+1)*scale, binVal), (255,255))
    histImg = np.flipud(histImg)
    return histImg

def calcHistSeries(series_, ranges=RMS_BIN_RANGES, column=None):
    """
    calculate the histogram of given Pandas Serires

    Parameters
    ----------
    series_ : Pandas Series
        object to calculate Histogram fro
    ranges : array like
        Array of the histogram bin boundaries.

    Returns
    -------
    histDF : DataFrame of the hists
    """

    labels = []
    filts = []
    hists = np.zeros((len(series_), len(ranges)))
    nitems = len(ranges)
    log.debug("ranges, {}\n".format(str(ranges)) )

    for i in range(nitems):
        if i == nitems - 1:
            label = "{}~".format(ranges[i])
            filt  = lambda x, y=ranges[i]: (x >= y) & True
        else:
            label = "{}~{}".format(ranges[i], ranges[i+1])
            filt  = lambda x, y=ranges[i], z=ranges[i+1]: (x >= y) & (x < z)
        labels.append(label)
        filts.append(filt)
    log.debug(', '.join(labels))
    log.debug("{}".format(str([ff==filts[0] for ff in filts])) )

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
        log.debug("series_: {}".format(str(series.values )))
        for j, filt in enumerate(filts):
            hists[i, j] = len(series.ix[ series.apply(filt) ] )
            log.debug("{}: {}".format((labels[j], hists[i, j])) )
    histDF = pd.DataFrame(hists, columns=labels).astype('int32')
    histDF = histDF.transpose() # for barplot, column as range lables
    return histDF

def im_fft_amplitude_phase(im, freqshift=True, method=None, raw_amplitude=False):
    '''get image DFT amplitude & phase for the input image by fft'''
    if method is None:
        method = 'fft'
    if method == 'fft':
        fourierfunc = np.fft.fft2
    elif method == 'dft':
        sys.path.append((os.path.dirname(os.path.abspath(__file__)))+"/../signal")
        from filters import dft
        fourierfunc = dft
    if not freqshift:
    # gen transform matrix T, so fft spectrum is zero-centered
        sys.path.append((os.path.dirname(os.path.abspath(__file__)))+"/../signal")
        T = np.zeros(im.shape, im.dtype)
        nrows, ncols = im.shape
        for y in range(nrows):
            for x in range(ncols):
                T[y, x] = (-1)**(x+y)
        im = np.multiply(im, T)

    # fft, and rescale the amplitude
    imfft = fourierfunc(im)
    if freqshift:
        imfft = np.fft.fftshift(imfft)
    rawamplitude = np.abs(imfft)
    amplitude  = np.log(1+ rawamplitude) # refer to DIPum
    # amplitude  = 1 + np.log(amplitude) # DIP form
    phase = np.angle(imfft)
    if raw_amplitude:
        amplitude = rawamplitude
    return (amplitude, phase)

def power_ratio_in_cutoff_frequency(amplitude, D0):
    '''
    A way to compute the cutoff frequency loci for LPF

    the ratio is defined as:

        r = 100 * Sum( P{inside D0} ) / Sum( P{image power} )

    Parameters
    ----------
    amplitude : 2D array
        image frequency amplitude or
        fourier transform result as complex data type
    D0 : float
        cutoff frequency

    Returns:
    --------
    ratio : float
        image power ratio inside cutoff frequency
    '''
    if 'complex' in str(amplitude.dtype):
        power = ff * np.conjugate(ff)
    else:
        power = amplitude**2

    from FreqeuncyFlt import distance_map
    D = distance_map(amplitude.shape)
    mask = ~(D <= D0)
    power_m = np.ma.array(power, mask=mask)

    ratio = 100 * np.sum(power_m) / np.sum(power)
    return ratio
