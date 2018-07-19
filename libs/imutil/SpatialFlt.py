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

__all__ = ['GaussianFilter', 'LaplaceFilter', 'SobelFilter', 'PrewittFilter']

def GaussianFilter(shape):
    N, M = shape
    if N%2 == 0 or M%2 == 0:
        raise ValueError("shape size {} should be odd!\n".format(repr(shape)))
    dst = []
    colG = cv_gaussian_kernel(M, dtype=np.int32).reshape((1, M) )
    rowG = cv_gaussian_kernel(N, dtype=np.int32).reshape((N, 1) )
    colT = np.tile(colG, (N, 1))
    rowT = np.tile(rowG, (1, M))
    dst = colT*rowT
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
        dst = GaussianFilter(shape)

    cy, cx = (s//2 for s in shape)
    tolsum = np.sum(dst)
    dst[cy, cx] = dst[cy, cx] - tolsum
    return dst

def SobelFilter(shape=None, axis=0, Prewitt=False):
    '''
    generate Sobel Filter, firstly generate the 1D differential array, then
    multiply the Gaussian function

    Parameters
    ----------
    axis : 0 or 1
        The axis of x or y
        - `0`: x, column
        [[-1.  0.  1.]      [-1.  0.  1.] X     [1]
         [-2.  0.  2.]                          [2]
         [-1.  0.  1.]]                         [1]
        - `1`: y, row
        [[-1. -2. -1.]
         [ 0.  0.  0.]
         [ 1.  2.  1.]]

    '''
    if shape is None:
        shape = (3, 3)
    N, M = shape
    if N%2 == 0 or M%2 == 0:
        raise ValueError("shape size {} should be odd!\n".format(repr(shape)))
    cy, cx = (s//2 for s in shape)
    if axis == 0:
        sobel_flt = np.ones(M)
        sobel_flt[0:cx] = -1
        sobel_flt[cx] = 0
        if Prewitt:
            sobel_smooth = np.ones(N)
        else:
            sobel_smooth = cv_gaussian_kernel(N, dtype=np.int32)
        colT = sobel_flt.reshape((1, M) )
        rowT = sobel_smooth.reshape((N, 1) )
    else:
        sobel_flt = np.ones(N)
        sobel_flt[0:cy] = -1
        sobel_flt[cy] = 0
        if Prewitt:
            sobel_smooth = np.ones(M)
        else:
            sobel_smooth = cv_gaussian_kernel(M, dtype=np.int32)

        colT = sobel_smooth.reshape((1, M) )
        rowT = sobel_flt.reshape((N, 1) )
    dst = np.matmul(rowT, colT)
    return dst

def PrewittFilter(shape=None, axis=0):
    return SobelFilter(shape=shape, axis=axis, Prewitt=True)