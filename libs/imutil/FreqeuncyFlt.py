# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-10 16:37:18

Frequency Domain Filters, so basically will have same shape with image

Last Modified by: ouxiaogu
"""

import numpy as np
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import padding_backward

__all__ = [ 'ILPF', 'IHPF', 'IBRF', 'IBPF',
            'BLPF', 'BHPF', 'BBRF', 'BBPF', 'BNRF', 'BNPF',
            'GLPF', 'GHPF', 'GBRF', 'GBPF',
            'HEF', 'LaplaceFilter',
            'imApplyFilter', 'distance_map',
            'HomomorphicFilter', 'imApplyHomomorphicFilter'
            ]

def distance_map(shape, notch=None):
    """distance = sqrt((x-cx)^2 + (y-cy)^2)"""
    N, M = shape
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    cx = (M - 1)/2
    cy = (N - 1)/2
    if notch is not None:
        cx, cy = notch
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
    z = z.astype(np.float32)
    return z

def IHPF(shape, D0):
    return 1 - ILPF(shape, D0)

def IBRF(shape, D0, W):
    D = distance_map(shape)
    z = np.logical_or(D > D0 + W/2, D < D0 - W/2)
    z = z.astype(np.float32)
    return z

def IBPF(shape, D0, W):
    return 1 - IBRF(shape, D0, W)

def BLPF(shape, D0, n):
    D = distance_map(shape)
    z = 1/(1 + np.power(D / D0, 2*n ))
    return z

def BHPF(shape, D0, n):
    return 1 - BLPF(shape, D0, n)

def BBRF(shape, D0, n, W):
    '''Butterworth Band Reject Filter'''
    D = distance_map(shape)

    z1 = np.power(D**2 - D0**2, 2*n)
    z2 = np.power(D*W, 2*n)
    z = z1/(z1 + z2)
    return z

def BBPF(shape, D0, n, W):
    return 1 - BBRF(shape, D0, n, W)

def BNRF(shape, notches=None, D0s=None, n=2, notch_origin='top-left'):
    '''
    Butterworth Notch Reject filter
    BNRF = ∏ H_i_{k} * H_i_{-k}

    Parameters
    ----------
    notches : array-like object
        each element is a pair of notch: (u, v), origin defined by
        `notch_origin`
    D0s : array-like object
        each element is a cut-off frequency
    n : int
        Butterworth filer order, default n = 2
    notch_origin : string, 'top-left' or 'center'
        - 'top-left', means input notch's origin is at top left of the image,
          need switch to image center to get the real (u, v)
    '''
    inline_origins = ['top-left', 'center']
    if len(notches) != len(D0s):
        raise ValueError("'notches' and 'D0s' have different length: {} & {}!\n".format(len(notches), len(D0s)))
    if notch_origin not in inline_origins:
        raise ValueError("input 'notch_origin' {} is not in supported list {}!\n".format(notch_origin, str(notch_origin)))
    N, M = shape
    if notch_origin == 'center':
        notches = [(u + (M - 1)/2, v + (N - 1)/2) for u, v in notches]
        inv_notches = [(-u, -v) for u, v in notches]
    elif notch_origin == 'top-left':
        inv_notches = [(M-1-u, N-1-v) for u, v in notches]
    z = np.ones(shape)
    for ix, notch in enumerate(notches):
        u, v = notch
        D0 = D0s[ix]
        D_p = distance_map(shape, (u, v))
        z_p = 1 - 1/(1 + np.power(D_p/D0, 2*n))

        D_n = distance_map(shape, inv_notches[ix])
        z_n = 1 - 1/(1 + np.power(D_n/D0, 2*n))
        z = z*z_p*z_n
    return z

def BNPF(shape, notches, D0s, n=2, notch_origin='top-left'):
    return 1 - BNRF(shape, notches, D0s, n, notch_origin)

def GLPF(shape, D0):
    D = distance_map(shape)
    z = np.exp(- D*D/(2 * D0**2) )
    return z

def GHPF(shape, D0):
    return 1 - GLPF(shape, D0)

def GBRF(shape, D0, W):
    D = distance_map(shape)
    z = 1 - np.exp(- ((D**2 - D0**2) / (D*W))**2 )
    return z

def GBPF(shape, D0, W):
    return 1 - GBRF(shape, D0, W)

def HEF(shape, HPFfunc=None, k1=0.5, k2=0.75, **kwargs):
    '''
    g(x, y) = f + k*(f - f_LP)
    G = F + k * (F - H_LP*F) = F + k * {(1- H_LP)*F} = (1 + k*H_HP)*F
    More generally, corresponding filter in Frequency domain is

    HEF = k1 + k2*HPF

    when k2 > k1, this is 'highboost filter', High-Emphasis Filtering(HEF),
    when k2 < k1, g(x, y) is so-called 'unsharp mask'

    Parameters
    ----------
    HPFfunc : function object
        specify the High Pass Filter function, default is GHPF
    k1 : float
        k1 control the distance to zero frequency wt plane, all the weight
        raw image frequency
    k2 : float
        k2 control the weight of high frequency
    kwargs: dict like object
        will pass into HPF functions
    '''
    if HPFfunc is None:
        HPFfunc = GHPF
    HPF = HPFfunc(shape, **kwargs)
    return k1 + k2*HPF

def LaplaceFilter(shape):
    '''
    Δf = ∇^2f <=> -4*pi^2*F, table 4.3 rule 12
    So Laplace Filter in frequency domain is:
    H(u, v) = -4*pi^2*D^2
    '''
    D = distance_map(shape)
    H = -4 * (np.pi)**2 * D**2
    return H

def HomomorphicFilter(shape, D0, gamma_L=0.25, gamma_H=2, c=1):
    '''
    Based on the illumination-reflection mode, consider an image as the
    product of illumination component(which usually is slow spatial variation)
    and reflection component (which usually is the big vary esp the junction
    between different objects).

    f(x, y) = i(x, y) * r(x, y)

    Homomorphic Filter:

    H = γL + (γH - γL) * (1 - e^{-c(D^2 / D0^2)} )

    Looks just like HFEF, regular spectrum plus a GHPF. However, it's totally
    different from regular filter, Homomorphic filter has its own way to apply.

    Reference
    ---------
    imApplyHomomorphicFilter
    '''
    D = distance_map(shape)
    z = gamma_L + (gamma_H - gamma_L) * (1 - np.exp(- c * D**2 / D0**2))
    return z

def imApplyFilter(src, fltFunc, **kwargs):
    '''
    perform general process to apply filters in frequency domain

    Parameters
    ----------
    src : 2D array
        input image
    fltFunc : function object
        specify the filter function
    kwargs : **dict
        key word argument to specify the shape of filter
    '''
    # 1. padding into 2X
    rawShape = src.shape
    fp = padding_backward(src, rawShape)

    # 2. matrix T=(-1)^(x+y) for fft shift in frequency domain
    N, M = fp.shape
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    T = (-1)**(x + y)
    fp = fp * T

    # 3. fft & filtering
    Fp = np.fft.fft2(fp)
    H = fltFunc(Fp.shape, **kwargs)
    Gp = Fp * H

    # 4. ifft, get real, revert translation T, and crop
    gp = np.fft.ifft2(Gp)
    gp = np.real(gp) * T
    N, M = rawShape
    g = gp[0:N, 0:M]
    return g

def imApplyHomomorphicFilter(src, D0, **kwargs):
    # 1. padding into 2X, add 1, natural log
    rawShape = src.shape
    src = src.astype(np.float32)
    fp = padding_backward(src, rawShape)
    fp = fp + 1
    fp_ln = np.log(fp)

    # 2. matrix T=(-1)^(x+y) for fft shift in frequency domain
    N, M = fp_ln.shape
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    T = (-1)**(x + y)
    fp_ln = fp_ln * T

    # 3. fft & filtering
    Fp_ln = np.fft.fft2(fp_ln)
    H = HomomorphicFilter(fp_ln.shape, D0, **kwargs)
    Gp_ln = Fp_ln * H

    # 4. ifft, get real, revert translation T, and crop
    gp_ln = np.fft.ifft2(Gp_ln)
    gp_ln = np.real(gp_ln) * T

    gp = np.exp(gp_ln)
    gp = gp - 1

    N, M = rawShape
    g = gp[0:N, 0:M]
    return g

if __name__ == '__main__':
    im = np.random.rand(9, 9)
    kwargs = {'D0': 3, 'n':2}
    a = imApplyFilter(im, BLPF, **kwargs)
    kwargs = {'n':2, 'D0': 3}
    b = imApplyFilter(im, BLPF, **kwargs)
    np.testing.assert_equal(a, b)
