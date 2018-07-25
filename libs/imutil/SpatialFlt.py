# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-18 17:59:20

Spatial Filters

Last Modified by: ouxiaogu
"""

import numpy as np
import cv2

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import cv_gaussian_kernel, padding, correlate
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../common")
import logger
log = logger.setup('SpatialFlt', level='info')

__all__ = ['GaussianFilter', 'LaplaceFilter', 'SobelFilter', 'PrewittFilter',
        'SOBEL_EDGE', 'SOBEL_SMOOTH', 'PREWITT_EDGE', 'PREWITT_SMOOTH',
        'SCHARR_EDGE', 'SCHARR_SMOOTH', 'LaplaceFilter3', 'BoxFilter',
        'LAPLACE_DIRECTIONAL', 'LAPLACE_POSITION', 'getDerivKernels',
        'ContraHarmonicMean', 'TrimedMean', 'kernelPreProc']

SOBEL_EDGE = np.array([1, 0, -1])
SOBEL_SMOOTH = np.array([1, 2, 1])
PREWITT_EDGE = np.array([1, 0, -1])
PREWITT_SMOOTH = np.array([1, 1, 1])
SCHARR_EDGE = np.array([1, 0, -1])
SCHARR_SMOOTH = np.array([3, 10, 3])
LAPLACE_DIRECTIONAL = np.array([1, -2, 1])
LAPLACE_POSITION = np.array([0, 1, 0])

def BoxFilter(shape, normalize=True):
    dst = np.ones(shape)
    if normalize:
        dst = dst / np.sum(dst)
    return dst

def GaussianFilter(shape, normalize=True):
    N, M = shape
    if N%2 == 0 or M%2 == 0:
        raise ValueError("shape size {} should be odd!\n".format(repr(shape)))
    dst = []
    colG = cv_gaussian_kernel(M, dtype=np.int32).reshape((1, M) )
    rowG = cv_gaussian_kernel(N, dtype=np.int32).reshape((N, 1) )
    colT = np.tile(colG, (N, 1))
    rowT = np.tile(rowG, (1, M))
    dst = colT*rowT
    if normalize:
        dst = dst / np.sum(dst)
    return dst

def LaplaceFilter(shape, gaussian=False):
    '''
    Δf = ∇^2f = f''x + f''y
    f''x = [f(x+1) - f(x)] - [f(x) - f(x-1)]
    if with Gaussian, it's so-called Laplacian of Gaussian (LoG):

        L = f(x, y)*g(x, y)   ΔL = Lxx + Lyy
    '''
    N, M = shape
    if N%2 == 0 or M%2 == 0:
        raise ValueError("shape size {} should be odd!\n".format(repr(shape)))

    dst = np.ones(shape)
    if gaussian:
        dst = GaussianFilter(shape, normalize=False)

    cy, cx = (s//2 for s in shape)
    tolsum = np.sum(dst)
    dst[cy, cx] = dst[cy, cx] - tolsum
    return dst

def LaplaceFilter3():
    N = M = 3
    L_x = np.matmul(LAPLACE_POSITION.reshape(N, 1), LAPLACE_DIRECTIONAL.reshape(1, M) )
    L_y = np.matmul(LAPLACE_DIRECTIONAL.reshape(N, 1), LAPLACE_POSITION.reshape(1, M)  )
    return L_x + L_y

def SobelFilter(shape=None, axis=0, Prewitt=False, dtype=np.int32, normalize=False):
    '''
    generate Sobel Filter, firstly generate the 1D differential array, then
    multiply the Gaussian function

    Parameters
    ----------
    axis : 0 or 1
        The axis of x or y
        - `0`: x, column, Sobel-x
         -1  0  1  =   [-1.  0.  1.] X  [1]
         -2  0  2                       [2]
         -1  0  1                       [1]
        because of convolve flipud filter, so what we input is [1.  0.  -1.]
        - `1`: y, row
        [[-1. -2. -1.]
         [ 0.  0.  0.]
         [ 1.  2.  1.]]
        because of convolve flipud filter, so what we input is [1.  0.  -1.]

    '''
    if shape is None:
        shape = (3, 3)
    N, M = shape
    if N%2 == 0 or M%2 == 0:
        raise ValueError("shape size {} should be odd!\n".format(repr(shape)))
    cy, cx = (s//2 for s in shape)
    if axis == 0:
        sobel_flt = np.ones(M)
        sobel_flt[cx:] = -1
        sobel_flt[cx] = 0
        if Prewitt:
            sobel_smooth = np.ones(N)
        else:
            sobel_smooth = cv_gaussian_kernel(N, dtype=dtype)
        if normalize:
            sobel_smooth = sobel_smooth/np.sum(sobel_smooth)
        colT = sobel_flt.reshape((1, M) )
        rowT = sobel_smooth.reshape((N, 1) )
    else:
        sobel_flt = np.ones(N)
        sobel_flt[cy:] = -1
        sobel_flt[cy] = 0
        if Prewitt:
            sobel_smooth = np.ones(M)
        else:
            sobel_smooth = cv_gaussian_kernel(M, dtype=dtype)
        if normalize:
            sobel_smooth = sobel_smooth/np.sum(sobel_smooth)
        colT = sobel_smooth.reshape((1, M) )
        rowT = sobel_flt.reshape((N, 1) )
    dst = np.matmul(rowT, colT)
    dst = dst.astype(dtype)
    return dst

def PrewittFilter(shape=None, axis=0):
    return SobelFilter(shape=shape, axis=axis, Prewitt=True)

def getDerivKernels(ktype, xorder, yorder, normalize=False):
    '''
    Get col / row separated kernels

    Parameters
    ----------
    ktype : string like
        kernel types
        - `Sobel`
        - `Prewitt`
        - `Scharr`
    xorder : int 0 or 1
        0: smooth; 1: edge
    yorder : int 0 or 1
        0: smooth; 1: edge
    normalize : bool
        whether to normalize the smooth filter

    Returns
    -------
    kx : 1D array-like
        kernel x
    ky : 1D array-like
        kernel y
    '''
    if ktype.lower() not in ['sobel', 'prewitt', 'scharr'] :
        raise ValueError("Input kernel type {} not in ['Sobel', 'Prewitt', 'Scharr']!\n".format(ktype))
    if xorder == 1 and yorder == 0:
        kx = eval(ktype.upper() + '_EDGE')
        ky = eval(ktype.upper() + '_SMOOTH')
        if normalize:
            ky = ky/np.sum(ky)
    elif xorder == 0 and yorder == 1:
        kx = eval(ktype.upper() + '_SMOOTH')
        ky = eval(ktype.upper() + '_EDGE')
        if normalize:
            kx = kx/np.sum(kx)
    else:
        raise ValueError("Input xorder {} yorder {} should one is 0, another is 1 !\n".format(xorder, yorder))
    return kx, ky

def ContraHarmonicMean(src, ksize=None, Q=1.5):
    '''
    Contra Harmonic filter for center pixel of current kernel:

    f'_c = g_k^(Q+1)/g_k^Q

    - Q > 0, good for pepper noise
    - Q < 0, good for salt noise
    - Q=0, Contra-harmonic Mean Filters = Arithmetic Mean Filters
    - Q=-1, Contra-harmonic Mean Filters = Harmonic Mean Filters

    if numsum=0, it means all non-negative pixel power sum is 0,
    only reason is this is a pure black flat area, should keep
    original pixel value;
    if numsum!=0, should have numsum>0, if just a pepper noise,
    densum increase quickly than numsum, then we can suppress or
    eliminate the pepper noise
    '''
    if ksize is None:
        ksize = (3, 3)
    elif np.ndim(ksize) == 0:
        ksize = (ksize, ksize)
    n, m = ksize
    fltX = np.full(m, 1.0)
    fltY = np.full(n, 1.0)

    src = src.astype(np.float64)
    d_eps = 1e-9
    addEps = False
    if np.min(src) == 0:
        log.debug("apply src add Eps\n")
        src += d_eps
        addEps = True
    denumerator = np.power(src, Q+1.)
    log.debug("denumerator:\n {}\n".format( str(denumerator)) )
    numerator = np.power(src, Q)
    log.debug("numerator:\n {}\n".format( str(numerator)) )
    densum = correlate(denumerator, fltX, fltY)
    log.debug("densum:\n {}\n".format( str(densum)) )
    numsum = correlate(numerator, fltX, fltY)
    log.debug("numsum:\n {}\n".format( str(numsum)) )
    dst = np.divide(densum, numsum, out=np.zeros_like(src), where=(numsum!=0))
    if addEps:
        log.debug("clear apply src add Eps\n")
        dst = np.subtract(dst, d_eps, out=dst, where=(dst>=d_eps))
    return dst

def kernelPreProc(src, ksize=None):
    '''
    Parameters
    ----------
    src : 2D array-like
        padded image array
    ksize : int or tuple
        kernel size

    Returns
    -------
    dst : 2D array-like
        dst, as zeros_like of the src
    gp : 2D array-like
        padded src image array
    N, M : int
        image #rows, #columns
    n, m : int
        kernel #rows, #columns, both should be odd
    hlFltSzY, hlFltSzX :
        half kernel size in rows(Y) and columns(X)
    '''
    if ksize is None:
        ksize = (3, 3)
    elif np.ndim(ksize) == 0:
        ksize = (ksize, ksize)
    N, M = src.shape
    n, m = ksize
    if n%2 == 0 or m%2 == 0:
        raise ValueError("ksize shape {} should be odd!\n".format(repr(ksize)))
    hlFltSzY, hlFltSzX = n//2, m//2
    gp = padding(src, (hlFltSzY, hlFltSzX) )
    dst = np.zeros_like(src)
    return dst, gp, N, M, n, m

def TrimedMean(src, ksize=None, d=0):
    '''alpha trimmed mean filter'''
    dst, gp, N, M, n, m = kernelPreProc(src, ksize=ksize)
    log.debug("dst\n {}\n".format( str(dst)) )
    log.debug("gp\n {}\n".format( str(gp)) )
    log.debug("{} {} {} {}\n".format( N, M, n, m) )
    if d >= n*m:
        raise ValueError("TrimedMean filter, removed elements number d {} should be less than kernel elements {}={}x{}!\n".format(d, n*m, n, m))
    hlTrim = d//2
    for r in range(N):
        for c in range(M):
            slices = gp[r:(r+n), c:(c+m)]
            arr = np.sort(slices, axis=None)
            if hlTrim > 0:
                arr = arr[hlTrim:-hlTrim]
            dst[r, c] = np.mean(arr)
    return dst

def adpM(src, ksize):


if __name__ == '__main__':
    print(eval('SOBEL_SMOOTH'))

    print(GaussianFilter((3, 3) ) )

    print(LaplaceFilter3() )
    print(LaplaceFilter((3, 3) ) )