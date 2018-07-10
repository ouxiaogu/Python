

"""
-*- coding: utf-8 -*-
Created: peyang, 2018-07-10 16:37:18

Frequency Domain Filters, so basically will have same shape with image

Last Modified by: peyang
"""

import numpy as np
import sys
import re

__all__ = [ 'ILPF', 'IHPF',
            'BLPF', 'BHPF',
            'GLPF', 'GHPF',
            'imApplyFilter']

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

def imApplyFilter(src, fltFunc, **kwargs):
    '''
    perform general process to apply filters in frequency domain
    
    Parameters
    ----------
    src : 2D array
        input image
    fltFunc : function object
        specify the filter function
    kwarg : *dict
        key word argument to specify the shape of filter
    '''
    # 1. padding into 2X
    sys.path.append("../signal")
    from filters import padding_backward 
    rawShape = src.shape
    fp = padding_backward(src, rawShape)

    # 2. matrix T=(-1)^(x+y) for fft shift in frequency domain
    M, N = fp.shape
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    T = (-1)**(x + y)
    fp = fp * T

    # 3. fft & filtering
    Fp = np.fft.fft2(fp)
    if 'D0' not in kwargs :
        D0 = min(M, N)/4
    else:
        D0 = kwargs['D0']

    if re.match('^B\w+PF', fltFunc.__name__):
        if 'n' not in kwargs :
            n = 2
        else:
            n = kwargs['n']
        H = fltFunc(Fp.shape, D0, n)
    else:
        H = fltFunc(Fp.shape, D0)
    Gp = Fp * H

    # 4. ifft, get real, revert translation T, and crop
    gp = np.fft.ifft2(Gp)
    gp = np.real(gp) * T
    N, M = rawShape
    g = gp[0:N, 0:M]
    return g



