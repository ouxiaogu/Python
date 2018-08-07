# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-16 10:29:47

Image intensity transform and spatial filter

Last Modified by: ouxiaogu
"""
import numpy as np
import math
from collections import OrderedDict

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import kernelPreProc
from ImDescriptors import Histogram, calcHist, cdfHisto

__all__ = ['equalizeHisto', 'specifyHisto',
        'localHistoEqualize1', 'localHistoEqualize2', 'localHistoEqualize3',
        'localHistoEqualize', 'normalize', 'intensityTransform', 'powerFunc',
        'imAdd', 'imSub', 'imMul', 'convertTo']

def powerFunc(c=1, gamma=0):
    '''
    power transformation function:
    s = T(r) = c * (s/255)^gamma * 255
    '''
    return lambda s: c*(s/255)**gamma * 255

def intensityTransform(src, mapping=None, Imax=255, dtype=None):
    '''
    image intensity transform

    Parameters
    ----------
    mapping : function, or list, or dict object
        support 3 modes:
        - `function`: input mapping as function
        - `list`: input mapping relation as a list,
        - `dict`: input mapping relation as a dict
    '''
    src = np.array(src)
    mfunc = None
    if callable(mapping):
        mfunc = lambda i: max(0, min(Imax, mapping(i) ) )
    elif isinstance(mapping, list) or isinstance(mapping, np.ndarray):
        if len(mapping) != 256:
            raise ValueError("intensityTransform, only support list type mapping with length=256, input mapping's is {}!\n".format(len(mapping)))
        mfunc = lambda i: max(0, min(Imax, mapping[i]))
    elif isinstance(mapping, dict) or isinstance(mapping, OrderedDict):
        kArr = np.array(list(mapping.keys()))
        if any(kArr < 0 ) or any(kArr > Imax):
            raise ValueError("intensityTransform, dict type mapping's key should in range of [0, {}]!\n".format(Imax))
        mfunc = lambda k: max(0, min(Imax, mapping[k]))
    dst = list(map(mfunc, src.flatten() ) )
    if np.ndim(src) == 0:
        dst = np.asscalar(np.array(dst) )
        if dtype is not None and callable(dtype):
            dst = dtype(dst)
    else:
        dst = np.array(dst).reshape(src.shape)
        if dtype is not None and callable(dtype):
            dst = dst.astype(dtype)
    return dst

def equalizeHisto(src, hist_type='list', Imax=255, dtype=None):
    '''
    equalize the histogram for input image

    Parameters
    ----------
    src : 2D image like
        current only grayscale image supported

    Returns
    -------
    dst : 2D image like:
        histogram-equalized result for src
    '''
    if src.dtype != np.uint8:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    if len(src.shape) != 2:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's shape is {}".format(repr(src.shape)))

    hist = calcHist(src, hist_type)
    mapping = cdfHisto(hist)

    dst = intensityTransform(src, mapping, Imax, dtype)
    return dst

def specifyHisto(src, ref, **kwargs):
    '''
    Histogram matching / specification
    Specify src's Histogram into ref's
    '''
    if src.dtype != ref.dtype or src.shape != ref.shape:
        raise ValueError("specifyHisto only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    if src.dtype != np.uint8:
        raise ValueError("specifyHisto only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    if len(src.shape) != 2:
        raise ValueError("specifyHisto only supports single channel grayscale image, src's shape is {}".format(repr(src.shape)))

    # 1. src: histogram and cdf
    sHist = calcHist(src)
    sCdf = cdfHisto(sHist)

    # 2. ref: histogram and cdf
    rHist = calcHist(ref)
    rCdf = cdfHisto(rHist)

    # 3. mapping function between src Cdf and ref Cdf
    # mapping between 2 rigid increase function, and with same range [0, 255]
    L = 256
    assert(len(sHist) == L)
    assert(len(rHist) == L)

    mapping = []
    rstart = 0
    for si in range(L):
        for ri in range(rstart, L):
            if ri == 0:
                if rCdf[ri] >= sCdf[si]:
                    break
            else:
                if (rCdf[ri] - sCdf[si])*(rCdf[ri - 1] - sCdf[si]) <= 0:
                    break
        mapping.append(ri)
        rstart = ri

    # 4. intensity level mapping
    dst = intensityTransform(src, mapping, **kwargs)
    return dst

def localHistoEqualize(src, ksize=None, **kwargs): # choose method 2
    return localHistoEqualize2(src, ksize, **kwargs)

def localHistoEqualize1(src, ksize=None): # 15.96s
    '''
    local Histogram matching / specification

    Parameters
    ----------
    src : 2D image like
        input image
    ksize : tuple like
        the shape of kernel to perform local histogram equalization

    Returns
    -------
    dst : 2D image like
        local histogram equalized image
    '''
    if src.dtype != np.uint8:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    dst, fp, N, M, n, m, cy, cx = kernelPreProc(src, ksize)

    for r in range(N):
        for c in range(M):
            slices = fp[r:(r+n), c:(c+m)]
            eqslice = equalizeHisto(slices, hist_type='OrderedDict') # ~17s
            # eqslice = equalizeHisto(slices) # 405.66s !!
            dst[r, c] = eqslice[cy, cx]
    return dst

def localHistoEqualize2(src, ksize=None, **kwargs): # 14.64s
    '''
    Based on localHistoEqualize1, only need to convert the center pixel of
    n*m kernel
    '''
    if src.dtype != np.uint8:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    dst, fp, N, M, n, m, cy, cx = kernelPreProc(src, ksize)

    for r in range(N):
        for c in range(M):
            slices = fp[r:(r+n), c:(c+m)]
            hist = calcHist(slices, hist_type='OrderedDict')
            cdf = cdfHisto(hist)
            cdfcenter = OrderedDict({slices[cx, cy]: cdf[slices[cx, cy]] } )
            dst[r, c] = intensityTransform(slices[cx, cy], mapping=cdfcenter, **kwargs)
    return dst

def localHistoEqualize3(src, ksize=None): # 24.33s
    '''
    Based on the h_new = h_old + h_datain - h_dataout
    '''
    if src.dtype != np.uint8:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    dst, fp, N, M, n, m, cy, cx = kernelPreProc(src, ksize)

    if 1: # 32.56s
        hist_tl = Histogram(calcHist(slices, hist_type='OrderedDict'))
        for r in range(N):
            if r == 0:
                hist_header = hist_tl
            else:
                hist_above = Histogram(calcHist(fp[r-1, 0:m], 'OrderedDict'))
                hist_below = Histogram(calcHist(fp[r+n-1, 0:m], 'OrderedDict'))
                hist_header = hist_header + hist_below - hist_above
            for c in range(M):
                slices = fp[r:(r+n), c:(c+m)]
                if c == 0:
                    hist = hist_header
                else:
                    hist_left = Histogram(calcHist(fp[r:(r+n), c-1], 'OrderedDict'))
                    hist_right = Histogram(calcHist(fp[r:(r+n), c+m-1], 'OrderedDict'))
                    hist = hist + hist_right - hist_left
                cdf = hist.cdf()
                cdfcenter = OrderedDict({slices[cx, cy]: cdf[slices[cx, cy]] } )
                dst[r, c] = intensityTransform(slices[cx, cy], mapping=cdfcenter)
    else: # 112.674s, much less than runtime of the 'list' type hist: 405.66s
        hist_tl = calcHist(slices)
        for r in range(N):
            if r == 0:
                hist_header = hist_tl
            else:
                hist_above = calcHist(fp[r-1, 0:m])
                hist_below = calcHist(fp[r+n-1, 0:m])
                hist_header = hist_header + hist_below - hist_above
            for c in range(M):
                slices = fp[r:(r+n), c:(c+m)]
                if c == 0:
                    hist = hist_header
                else:
                    hist_left = calcHist(fp[r:(r+n), c-1])
                    hist_right = calcHist(fp[r:(r+n), c+m-1])
                    hist = hist + hist_right - hist_left
                cdf = cdfHisto(hist)
                cdfcenter = OrderedDict({slices[cx, cy]: cdf[slices[cx, cy]] } )
                dst[r, c] = intensityTransform(slices[cx, cy], mapping=cdfcenter, **kwargs)
    return dst

def normalize(src, Imax=255, dtype=None):
    vmin = np.min(src)
    vmax = np.max(src)

    mfunc = lambda v: Imax * (v - vmin)/(vmax - vmin)
    dst = intensityTransform(src, mfunc, Imax=Imax, dtype=dtype)
    return dst

def convertTo(src, dtype=None, alpha=1, beta=0):
    '''s = dtype(alpha*r + beta)'''
    dst = dtype(src*alpha + beta)
    return dst

def imAdd(src, mask, Imax=None, sub=False):
    '''
    Image add here means, image enhancement
    There may be overflow for image Add and multiply, as the example below:

    In [164]: a = np.array([255, 255 ], dtype=np.uint8)

    In [165]: b = np.array([6, 7 ], dtype=np.uint8)

    In [166]: a + b
    Out[166]: array([5, 6], dtype=uint8)

    The solution here is, transform the dtype into np.float32 first,
    convert the dtype of src at the end
    '''
    # disable the dtype check, by use higher accuracy dtype firstly
    '''
    if src.dtype == mask.dtype:
        raise ValueError("imAdd, input images have different dtype, src: {}, mask: {}!\n".format(str(src), str(mask)))
    '''
    dtype = np.float64
    im = np.array(src, copy=True, dtype=dtype)
    mask = mask.astype(dtype)
    if not sub:
        dst = im + mask
    else:
        dst = im - mask
    if Imax is None:
        sys.stderr.write("imAdd, intensity upper bound Imax is not given\n")
    elif Imax > 0:
        dst = np.clip(dst, 0, Imax)
    return dst.astype(src.dtype)

def imSub(src, mask, Imax=None):
    return imAdd(src, mask, Imax=Imax, sub=True)

def imMul(src, mask, Imax=None):
    '''image multiply usually is for src*mask'''
    dtype = np.float64
    im = np.array(src, copy=True, dtype=dtype)
    mask = mask.astype(dtype)
    dst = im * mask
    if Imax is None:
        sys.stderr.write("imAdd, intensity upper bound Imax is not given\n")
    elif Imax > 0:
        dst = normalize(dst, Imax=Imax)
    return dst.astype(src.dtype)