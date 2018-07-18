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

__all__ = ['GaussianFilter', 'LaplaceFilter']

def GaussianFilter(shape):
    N, M = shape
    if N%2 == 0 or M%2 == 0:
        raise ValueError("shape size {} should be odd!\n".format(repr(shape)))
    dst = []
    colG = cv_gaussian_kernel(M, dtype=np.int32).reshape((1, M) )
    rowG = cv_gaussian_kernel(N, dtype=np.int32).reshape((N, 1) )
    try:
        colT = np.tile(colG, [N, 1])
        rowT = np.tile(rowG, [1, M])
    except ValueError:
        print(colG)
        print(rowG)
        raise
    dst = colT*rowT
    return dst

def LaplaceFilter(shape, gaussian=False):
    '''
    Δf = ∇^2f = f''x + f''y
    f''x = [f(x+1) - f(x)] - [f(x) - f(x-1)]
    '''
    N, M = shape
    if N%2 == 0 or M%2 == 0:
        raise ValueError("shape size {} should be odd!\n".format(repr(shape)))

    dst = np.ones(shape)
    if gaussian:
        dst = GaussianFilter(shape)

    cx, cy = (s//2 for s in shape)
    tolsum = np.sum(dst)
    dst[cx, cy] = dst[cx, cy] - tolsum
    return dst