# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-16 10:29:47

Image intensity transform and spatial filter

Last Modified by: ouxiaogu
"""
import numpy as np
import math
import cv2

__all__ = ['Histogram', 'equalizeHisto', 'specifyHistoo']

def Histogram(src):
    '''
    histogram for grayscale image, intensity level is 256
    '''
    src = np.array(src)
    if src.dtype != np.uint8:
        raise ValueError("Histogram only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    if len(src.shape) != 2:
        raise ValueError("Histogram only supports single channel grayscale image, src's shape is {}".format(repr(src.shape)))

    hist = np.zeros(256, dtype=np.int32)
    N, M = src.shape
    for r in range(N):
        for c in range(M):
            hist[ src[r, c] ] += 1
    return hist


def equalizeHisto(src):
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

    hist = Histogram(src)
    mapping = cdfHisto(hist)

    map_func = lambda i: mapping[i]
    dst = list(map(map_func, src.flatten() ) )
    dst = np.array(dst, dtype=np.uint8).reshape(src.shape)
    return dst

def cdfHisto(hist):
    L = 256
    cdf_func = lambda x: max(0, min(255, int(math.ceil((L-1) * x / np.sum(hist) + 0.5) ) ) )
    mapping = []
    cumsum = 0
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
    sCdf = cdfHisto(src)

    # 2. ref: histogram and cdf
    rHist = Histogram(ref)
    rCdf = cdfHisto(ref)

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
    map_func = lambda i: mapping[i]
    dst = list(map(map_func, src.flatten() ) )
    dst = np.array(dst, dtype=np.uint8).reshape(src.shape)
    return dst