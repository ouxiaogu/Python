# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-16 10:29:47

Image intensity transform and spatial filter

Last Modified by: ouxiaogu
"""
import numpy as np
import math
import cv2
from collections import OrderedDict

import sys
sys.path.append("../signal")
from filters import padding

__all__ = ['Histogram', 'equalizeHisto', 'specifyHisto', 'cdfHisto', 'localHistoEqualize']

def intensityTransfrom(src, mapping=None):
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
    mfunc = None
    if callable(mapping):
        mfunc = lambda i: max(0, min(255, mapping(i) ) )
    elif isinstance(mapping, list):
        if len(mapping) != 256:
            raise ValueError("intensityTransfrom, only support list type mapping with length=256, input mapping's is {}!\n".format(len(mapping)))
        mfunc = lambda i: max(0, min(255, mapping[i]))
    elif isinstance(mapping, dict) or isinstance(mapping, OrderedDict):
        kArr = np.array(list(mapping.keys()), dtype=np.uint8)
        if any(kArr < 0 ) or any(kArr > 255):
            raise ValueError("intensityTransfrom, dict type mapping's key should in range of [0, 255]!\n")
        mfunc = lambda k: max(0, min(255, mapping[k]))
    dst = list(map(mfunc, src.flatten() ) )
    dst = np.array(dst, dtype=np.uint8).reshape(src.shape)
    return dst

def Histogram(src, hist_type='list'):
    '''
    histogram for grayscale image, intensity level is 256

    Parameters
    ----------
    hist_type : string like
        two types of hist_type to define histogram
        - 'list': return hist as length=256 list, `hist[i]` to store the
          times of intensity `i` occurs
        - 'OrderedDict': return hist as length<=256 dict, `hist[k]` to store
          the times of intensity `k` occurs, and will convert to OrderedDict
          and sorted, for the sake of cdfHisto computation
    '''
    src = np.array(src)
    if src.dtype != np.uint8:
        raise ValueError("Histogram only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    if len(src.shape) != 2:
        raise ValueError("Histogram only supports single channel grayscale image, src's shape is {}".format(repr(src.shape)))

    N, M = src.shape
    if hist_type == 'list':
        hist = np.zeros(256, dtype=np.int32)
        for r in range(N):
            for c in range(M):
                hist[ src[r, c] ] += 1
    elif hist_type == 'OrderedDict':
        hist = {}
        for r in range(N):
            for c in range(M):
                if src[r, c] not in hist.keys():
                    hist[src[r, c] ] = 0
                hist[src[r, c]] += 1
        hist = OrderedDict(sorted(hist.items(), key=lambda kv: kv[0]))
    return hist

def equalizeHisto(src, hist_type='list'):
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

    hist = Histogram(src, hist_type)
    mapping = cdfHisto(hist)

    dst = intensityTransfrom(src, mapping)
    return dst

def cdfHisto(hist):
    '''
    hist: list with length = 256
    cdf: cumulative distribution function
    mapping: convert cdf into 0-255 grayscale levels
    '''
    L = 256
    cumsum = 0
    if isinstance(hist, OrderedDict):
        mapping = OrderedDict()
        cdf_func = lambda x: int(math.floor((L-1) * x / np.sum(list(hist.values() ) ) + 0.5) )
        for k, v in hist.items():
            cumsum += v
            mapping[k] = cdf_func(cumsum)
    else:
        mapping = []
        cdf_func = lambda x: int(math.floor((L-1) * x / np.sum(hist) + 0.5) )
        for v in hist:
            cumsum += v
            mapping.append(cdf_func(cumsum) )
    return mapping

def specifyHisto(src, ref):
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
    sHist = Histogram(src)
    sCdf = cdfHisto(sHist)

    # 2. ref: histogram and cdf
    rHist = Histogram(ref)
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
    dst = intensityTransfrom(src, mapping)
    return dst

def localHistoEqualize(src, ksize=None):
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
    if len(src.shape) != 2:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's shape is {}".format(repr(src.shape)))
    if ksize is None:
        ksize = (3, 3)
    try:
        if(len(ksize) != 2):
            raise ValueError("ksize dimension {} should be the same with image's {}!\n".format(len(ksize), len(im.shape)))
    except:
        ksize = (ksize, ksize)

    n, m = ksize
    if n%2 == 0 or m%2 == 0:
        raise ValueError("ksize shape {} should be odd!\n".format(repr(ksize)))
    padshape = tuple(sz//2 for sz in ksize)
    cx, cy = padshape
    fp = padding(src, padshape)
    dst = np.zeros(src.shape, src.dtype)

    N, M = src.shape
    for r in range(N):
        for c in range(M):
            slices = fp[r:(r+n), c:(c+m)]
            eqslice = equalizeHisto(slices, hist_type='OrderedDict') # ~17s
            # eqslice = equalizeHisto(slices) # 405.66s !!
            dst[r, c] = eqslice[cy, cx]
    return dst