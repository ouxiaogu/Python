# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-18 17:59:20

Spatial Filters

Last Modified by: ouxiaogu
"""

import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import cv_gaussian_kernel

__all__ = ['GaussianFilter', 'LaplaceFilter', 'SobelFilter', 'PrewittFilter',
        'SOBEL_EDGE', 'SOBEL_SMOOTH', 'PREWITT_EDGE', 'PREWITT_SMOOTH',
        'SCHARR_EDGE', 'SCHARR_SMOOTH', 'LaplaceFilter3', 'BoxFilter',
        'LAPLACE_DIRECTIONAL', 'LAPLACE_POSITION', 'getDerivKernels'
        ]

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

if __name__ == '__main__':
    print(eval('SOBEL_SMOOTH'))

    print(GaussianFilter((3, 3) ) )

    print(LaplaceFilter3() )
    print(LaplaceFilter((3, 3) ) )