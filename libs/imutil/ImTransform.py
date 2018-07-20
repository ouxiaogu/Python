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
from filters import padding

__all__ = ['calcHist', 'equalizeHisto', 'specifyHisto', 'cdfHisto',
        'localHistoEqualize1', 'localHistoEqualize2', 'localHistoEqualize3',
        'addOrdereddDict', 'subOrdereddDict', 'Histogram',
        'localHistoEqualize', 'normalize', 'intensityTransform', 'powerFunc',
        'imAdd', 'imMul']

class Histogram(object):
    """docstring for Histogram"""
    def __init__(self, hist, **kwargs):
        super(Histogram, self).__init__()
        self.hist = hist

    def _validate_args(self, rhs):
        if not isinstance(rhs, Histogram):
            raise TypeError("rhs {} is not the Histogram object!\n".format(type(rhs) ))
        if type(self.hist) != type(rhs.hist):
            raise TypeError("Histogram self {} and rhs {} is not in the same type!\n".format(type(self.hist), type(rhs.hist) ))

    def __add__(self, rhs):
        self._validate_args(rhs)
        if isinstance(self.hist, OrderedDict):
            hist = addOrdereddDict(self.hist, rhs.hist)
        elif isinstance(self.hist, list):
            hist = (np.array(self.hist) + np.array(rhs.hist)).tolist()
        return Histogram(hist)

    def __sub__(self, rhs):
        self._validate_args(rhs)
        if isinstance(self.hist, OrderedDict):
            hist = subOrdereddDict(self.hist, rhs.hist)
        elif isinstance(self.hist, list):
            hist = (np.array(self.hist) - np.array(rhs.hist)).tolist()
        return Histogram(hist)

    def cdf(self):
        return cdfHisto(self.hist)

def cdfHisto(hist, normalized=True):
    '''
    hist: list with length = 256
    cdf: cumulative distribution function
    mapping: convert cdf into 0-255 grayscale levels
    '''
    L = 256
    cumsum = 0
    if isinstance(hist, OrderedDict):
        mapping = OrderedDict()
        cumsum = np.cumsum(list(hist.values() ) )
        hist = list(hist.items())
        if normalized:
            tolsum = cumsum[-1]
            cdf_func = lambda x: int(math.floor((L-1) * x / tolsum  + 0.5) )
            for i, kv in enumerate(hist):
                mapping[kv[0]] = cdf_func(cumsum[i])
        else:
            for i, kv in enumerate(hist):
                mapping[kv[0]] = cumsum[i]
    else:
        cumsum = np.cumsum(hist)
        if normalized:
            tolsum = cumsum[-1]
            cdf_func = lambda x: int(math.floor((L-1) * x / tolsum + 0.5) )
            mapping = [cdf_func(s) for s in cumsum]
        else:
            mapping = cumsum

    return mapping

def addOrdereddDict(lhs, rhs):
    lhs = list(lhs.items() )
    rhs = list(rhs.items() )

    dst = []
    l = r = 0
    while True:
        if r == len(rhs):
            if l < len(lhs) :
                dst += lhs[l:]
            break
        elif l == len(lhs):
            if r < len(rhs) :
                dst += rhs[r:]
            break
        else:
            if lhs[l][0] < rhs[r][0]:
                dst.append((lhs[l][0], lhs[l][1]) )
                l += 1
            elif lhs[l][0] == rhs[r][0]:
                dst.append((lhs[l][0], lhs[l][1] + rhs[r][1]) )
                l += 1
                r += 1
            elif lhs[l][0] > rhs[r][0]:
                dst.append((rhs[r][0], rhs[r][1]) )
                r += 1
    return  OrderedDict(dst)

def subOrdereddDict(lhs, rhs):
    dst = []
    l = r = 0
    for rk, rv in rhs.items():
        if rk not in lhs.keys():
            raise KeyError("subOrdereddDict, rhs key {} is not in lhs!\n".format(rk))
        lhs[rk] -= rhs[rk]
    return lhs

def powerFunc(c=1, gamma=0):
    '''
    power transformation function:
    s = T(r) = c * (s/255)^gamma * 255
    '''
    return lambda s: c*(s/255)**gamma * 255

def intensityTransform(src, mapping=None, dtype=None):
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
        mfunc = lambda i: max(0, min(255, mapping(i) ) )
    elif isinstance(mapping, list):
        if len(mapping) != 256:
            raise ValueError("intensityTransform, only support list type mapping with length=256, input mapping's is {}!\n".format(len(mapping)))
        mfunc = lambda i: max(0, min(255, mapping[i]))
    elif isinstance(mapping, dict) or isinstance(mapping, OrderedDict):
        kArr = np.array(list(mapping.keys()))
        if any(kArr < 0 ) or any(kArr > 255):
            raise ValueError("intensityTransform, dict type mapping's key should in range of [0, 255]!\n")
        mfunc = lambda k: max(0, min(255, mapping[k]))
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

def calcHist(src, hist_type='list'):
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

    sz = src.size
    arr = src.flatten()
    if hist_type == 'list':
        hist = np.zeros(256, dtype=np.int32)
        for i in range(sz):
            hist[ arr[i] ] += 1
    elif hist_type == 'OrderedDict':
        hist = {}
        for i in range(sz):
            if arr[i] not in hist.keys():
                hist[arr[i] ] = 0
            hist[arr[i] ] += 1
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

    hist = calcHist(src, hist_type)
    mapping = cdfHisto(hist)

    dst = intensityTransform(src, mapping)
    return dst

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
    dst = intensityTransform(src, mapping)
    return dst

def localHistoEqualize(src, ksize): # choose method 2
    return localHistoEqualize2(src, ksize)

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
    if len(src.shape) != 2:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's shape is {}".format(repr(src.shape)))
    if ksize is None:
        ksize = (3, 3)
    try:
        if(len(ksize) != 2):
            raise ValueError("ksize dimension {} should be the same with image's {}!\n".format(len(ksize), len(src.shape)))
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

def localHistoEqualize2(src, ksize=None): # 14.64s
    '''
    Based on localHistoEqualize1, only need to convert the center pixel of
    n*m kernel
    '''
    if src.dtype != np.uint8:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    if len(src.shape) != 2:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's shape is {}".format(repr(src.shape)))
    if ksize is None:
        ksize = (3, 3)
    try:
        if(len(ksize) != 2):
            raise ValueError("ksize dimension {} should be the same with image's {}!\n".format(len(ksize), len(src.shape)))
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
            hist = calcHist(slices, hist_type='OrderedDict')
            cdf = cdfHisto(hist)
            cdfcenter = OrderedDict({slices[cx, cy]: cdf[slices[cx, cy]] } )
            dst[r, c] = intensityTransform(slices[cx, cy], mapping=cdfcenter)
    return dst

def localHistoEqualize3(src, ksize=None): # 24.33s
    '''
    Based on the h_new = h_old + h_datain - h_dataout
    '''
    if src.dtype != np.uint8:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's dtype is {}".format(repr(src.dtype)))
    if len(src.shape) != 2:
        raise ValueError("equalizeHisto only supports single channel grayscale image, src's shape is {}".format(repr(src.shape)))
    if ksize is None:
        ksize = (3, 3)
    try:
        if(len(ksize) != 2):
            raise ValueError("ksize dimension {} should be the same with image's {}!\n".format(len(ksize), len(src.shape)))
    except:
        ksize = (ksize, ksize)

    n, m = ksize
    if n%2 == 0 or m%2 == 0:
        raise ValueError("ksize shape {} should be odd!\n".format(repr(ksize)))
    padshape = tuple(sz//2 for sz in ksize)
    cx, cy = padshape
    fp = padding(src, padshape)
    dst = np.zeros(src.shape, src.dtype)

    # initial image slices
    slices = fp[0:n, 0:m]
    N, M = src.shape

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
                dst[r, c] = intensityTransform(slices[cx, cy], mapping=cdfcenter)
    return dst

def normalize(src, Imax=255, dtype=None):
    vmin = np.min(src)
    vmax = np.max(src)

    mfunc = lambda v: Imax * (v - vmin)/(vmax - vmin)
    dst = intensityTransform(src, mfunc, dtype)
    return dst

def imAdd(src, mask, Imax=None):
    '''
    Image add here means, image enhancement
    There may be overflow for image Add and multiply, as the example below:

    In [164]: a = np.array([255, 255 ], dtype=np.uint8)

    In [165]: b = np.array([6, 7 ], dtype=np.uint8)

    In [166]: a + b
    Out[166]: array([5, 6], dtype=uint8)

    The solution here is, transform the dtype into np.float64 first,
    convert the dtype of src at the end
    '''
    # disable the dtype check, by use higher accuracy dtype firstly
    '''
    if src.dtype == mask.dtype:
        raise ValueError("imAdd, input images have different dtype, src: {}, mask: {}!\n".format(str(src), str(mask)))
    '''
    src = src.astype(np.float64)
    mask = mask.astype(np.float64)
    dst = src + mask

    if Imax is None:
        sys.stderr.write("imAdd, intensity upper bound Imax is not given")
    elif Imax > 0:
        dst = np.clip(dst, 0, Imax)
    return dst.astype(src.dtype)

def imMul(src, mask, Imax=None):
    '''image multiply usually is for src*mask'''
    src = src.astype(np.float64)
    mask = mask.astype(np.float64)
    dst = src * mask
    if Imax is None:
        sys.stderr.write("imAdd, intensity upper bound Imax is not given")
    elif Imax > 0:
        dst = normalize(dst, Imax=Imax)
    return dst.astype(src.dtype)