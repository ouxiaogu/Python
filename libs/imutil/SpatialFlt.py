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

from ImTransform import imSub
from ImDescriptors import getImageInfo

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import cv_gaussian_kernel, correlate, kernelPreProc, fltGenPreProc, applySepFilter, applyKernelOperator, padding

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../common")
import logger
log = logger.setup('SpatialFlt', level='info')

__all__ = ['GaussianFilter', 'LaplaceFilter', 'SobelFilter', 'PrewittFilter',
        'SOBEL_EDGE', 'SOBEL_SMOOTH', 'PREWITT_EDGE', 'PREWITT_SMOOTH',
        'SCHARR_EDGE', 'SCHARR_SMOOTH', 'LaplaceFilter3', 'BoxFilter',
        'LAPLACE_DIRECTIONAL', 'LAPLACE_POSITION', 'getDerivXYKernel',
        'getMeanXYKernel', 'ContraHarmonicMean', 'TrimedMean',
        'adpMean', 'adpMedian']

SOBEL_EDGE = np.array([1, 0, -1])
SOBEL_SMOOTH = np.array([1, 2, 1])
PREWITT_EDGE = np.array([1, 0, -1])
PREWITT_SMOOTH = np.array([1, 1, 1])
SCHARR_EDGE = np.array([1, 0, -1])
SCHARR_SMOOTH = np.array([3, 10, 3])
LAPLACE_DIRECTIONAL = np.array([1, -2, 1])
LAPLACE_POSITION = np.array([0, 1, 0])

def BoxFilter(shape, normalize=True):
    fltGenPreProc(shape)
    dst = np.ones(shape)
    if normalize:
        dst = dst / np.sum(dst)
    return dst

def GaussianFilter(shape, normalize=True, **kwargs):
    N, M, _, _ = fltGenPreProc(shape)

    dst = []
    colG = cv_gaussian_kernel(M, **kwargs).reshape((1, M) )
    rowG = cv_gaussian_kernel(N, **kwargs).reshape((N, 1) )
    # colT = np.tile(colG, (N, 1))
    # rowT = np.tile(rowG, (1, M))
    # dst = colT*rowT
    dst = np.matmul(rowG, colG)
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
    N, M, cx, cy = fltGenPreProc(shape)

    dst = np.ones(shape)
    if gaussian:
        dst = GaussianFilter(shape, normalize=False)

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
    N, M, cy, cx = fltGenPreProc(shape)
    if axis == 'x':
        axis = 0
    elif axis == 'y':
        axis = 1
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

def getDerivXYKernel(ktype, xorder, yorder, normalize=False):
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
    DeriviateKernels = ['sobel', 'prewitt', 'scharr']
    if ktype.lower() not in DeriviateKernels :
        raise ValueError("Input kernel type {} not in {}!\n".format(ktype, str(DeriviateKernels)))
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

def getMeanXYKernel(ksize=None, ktype='box', normalize=True, **kwargs):
    MeanFilter = ['box', 'gaussian']
    if ktype.lower() not in MeanFilter :
        raise ValueError("Input kernel type {} not in {}!\n".format(ktype, str(MeanFilter)))
    N, M, _, _ = fltGenPreProc(ksize)
    if ktype == 'box':
        ky = np.ones(N)
        kx = np.ones(M)
    elif ktype == 'gaussian':
        ky = cv_gaussian_kernel(N, **kwargs)
        kx = cv_gaussian_kernel(M, **kwargs)
    if normalize:
        ky = ky/np.sum(ky)
        kx = kx/np.sum(kx)
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

def TrimedMean(src, ksize=None, d=0):
    '''alpha trimmed mean filter'''
    dst, gp, N, M, n, m, _, _ = kernelPreProc(src, ksize=ksize)
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

def applyMeanFilter(src, ksize=3, ktype='box', **kwargs):
    kernelPreProc(src, ksize=ksize)
    kx, ky = getMeanXYKernel(ksize=ksize, ktype=ktype, normalize=True, **kwargs) # always normalize in getMeanXYKernel
    dst = applySepFilter(src, kx, ky)
    return dst

def adpMean(src, ksize, noise=None, noise_var=0, Imax=255, **kwargs):
    '''
    adaptive, local noise reduction filter

    f' = g - VarN/VarL (g - mL)

    Where,
        g : degraded image
        VarL :  Variance of `g` within local kernel
        VarN :  Variance of `noise` within local kernel
        mL :  smoothed image of `g` within local kernel
        f' : restored image

    1. VarN = 0, f' = g
    2. VarL >> VarN,  f' = g, kept high contrast image areas
    3. VarL ~= VarN,   = , reduce noise by smooth

    Parameters
    ----------
    noise : 2D image like
        should have the same size like the src image or padded image size
    noise_var : double
        constant Variance of the noise image, (std)^2
    '''
    dtype = np.float64
    g = np.array(src, copy=True, dtype=dtype)
    mL = applyMeanFilter(g, ksize, 'box', **kwargs)
    VarL = applyKernelOperator(g, ksize, np.var)
    if noise is not None:
        if any(np.array(src.shape) - np.array(noise.shape) != 0):
            raise ValueError("adpMean, src {} and noise {} image are not in the same shape!\n".format(str(src.shape), str(noise.shape)))
        VarN = applyKernelOperator(noise, ksize, np.var)
    else:
        noise_var = dtype(noise_var)
        VarN = np.full(g.shape, noise_var)
    wt = np.divide(VarN, VarL, out=np.zeros_like(VarN), where=(VarL!=0))
    wt = np.clip(wt, 0, 1)
    log.debug('g: ' + getImageInfo(g) )
    log.debug('mL: ' + getImageInfo(mL) )
    log.debug('VarL: ' + getImageInfo(VarL) )
    log.debug('VarN: ' + getImageInfo(VarN) )
    log.debug('wt: ' + getImageInfo(wt) )
    dst = imSub(g, wt*(g - mL), Imax=255)
    return dst

def adpMedian(src, ksize=3, max_ksize=7):
    '''
    adaptive median filter
    '''
    dst, _, N, M, n, m, _, _ = kernelPreProc(src, ksize)
    gp_max = padding(src, max_ksize//2)

    operators = [np.min, np.max, np.median]
    Lmin, Lmax, Lmed = applyKernelOperator(src, ksize, operators)

    for r in range(N):
        for c in range(M):
            dst[r, c] = adpMedianPixel(gp_max, (r, c), src[r,c], ksize, max_ksize, Lmin[r,c], Lmax[r,c], Lmed[r,c])
    return dst

def adpMedianPixel(gp_max, coord, intensity, ksize, max_ksize, lmin, lmax, lmed):
    if lmed > lmin and  lmed < lmax:
        if intensity > lmin and intensity < lmax:
            return intensity
        else:
            return lmed
    else:
        ksize += 2
        if ksize > max_ksize:
            return lmed
        r, c = coord
        shift = max_ksize//2
        cx = cy = ksize//2
        slices = gp_max[(r+shift-cy):(r+shift+cy+1), (c+shift-cx):(c+shift+cx+1)]
        lmin = np.min(slices)
        lmax = np.max(slices)
        lmed = np.median(slices)
        return adpMedianPixel(gp_max, coord, intensity, ksize, max_ksize, lmin, lmax, lmed)

if __name__ == '__main__':
    print(eval('SOBEL_SMOOTH'))

    print(GaussianFilter((3, 3) ) )

    print(LaplaceFilter3() )
    print(LaplaceFilter((3, 3) ) )