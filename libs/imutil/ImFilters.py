"""
-*- coding: utf-8 -*-
Created: peyang, 2018-07-10 16:37:18

Frequency Domain Filters, so basically will have same shape with image

Last Modified by: peyang
"""

import numpy as np

__all__ = [ 'ILPF', 'IHPF',
            'BLPF', 'BHPF',
            'GLPF', 'GHPF']

def distance_map(shape):
    """distance = sqrt((x-cx)^2 + (y-cy)^2)"""
    N, M = shape
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    cx = (M - 1)/2
    cy = (N - 1)/2
    z = np.sqrt((x - cx)**2 + (y - cy)**2)
    return z

def ILPF(shape, D0):
    """
    Ideal low pass filter
    1 if D(u, v) <= D0

    Parameters:
    -----------
    shape : tuple
        dst shape
    D0 : double
        cut-off frequency

    Returns
    -------
    dst : 2D array
        dst shape will be as given
    """
    D = distance_map(shape)
    z = (D <= D0)
    z = z.astype(float)
    return z

def IHPF(shape, D0):
    return 1 - ILPF(shape, D0)

def BLPF(shape, D0, n):
    D = distance_map(shape)
    z = 1/(1 + np.power(D / D0, 2*n ))
    return z

def BHPF(shape, D0, n):
    return 1 - BLPF(shape, D0, n)

def GLPF(shape, D0):
    D = distance_map(shape)
    z = np.exp(- D*D/(2 * D0**2) )
    return z

def GHPF(shape, D0):
    return 1 - GLPF(shape, D0)