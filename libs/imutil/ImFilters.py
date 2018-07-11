# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-10 16:37:18

Frequency Domain Filters, so basically will have same shape with image

Last Modified by: peyang
"""

import numpy as np
import sys
import re

__all__ = [ 'ILPF', 'IHPF', 'IBRF', 'IBPF',
            'BLPF', 'BHPF', 'BBRF', 'BBPF',
            'GLPF', 'GHPF', 'GBRF', 'GBPF',
            'HFEF', 'LaplaceFilter',
            'imApplyFilter', 'distance_map',
            'HomomorphicFilter', 'imApplyHomomorphicFilter'
            ]

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
    D = distance_map(shape)
    z1 = np.power(D**2 - D0**2, 2*n)
    z2 = np.power(D*W, 2*n)
    z = z1/(z1 + z2)
    return z

def BBPF(shape, D0, n, W):
    return 1 - BBRF(shape, D0, n, W)

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

def HFEF(shape, HPFfunc, k1=0.5, k2=0.75, *args):
    '''
    g(x, y) = f + k*(f - f_LP)
    G = F + k * (F - H_LP*F) = F + k * {(1- H_LP)*F} = (1 + k*H_HP)*F
    when k2 > k1, this is 'highboost filter', corresponding filter in
    Frequency domain is, High Frequency Emphasis Filter(HFEF), as:
    HFEF = k1 + k2*HPF
    when k2 < k1, g(x, y) is so-called 'unsharp mask'

    Parameters
    ----------
    HPFfunc : function object
        specify the High Pass Filter function
    k1 : float
        k1 control the distance to zero frequency wt plane, all the weight
        raw image frequency
    k2 : float
        k2 control the weight of high frequency
    args: list like object
        will pass into HPF functions in order
    '''
    HPF = HPFfunc(shape, *args)
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
    sys.path.append("../signal")
    from filters import padding_backward
    rawShape = src.shape
    fp = padding_backward(src, rawShape)

    # 2. matrix T=(-1)^(x+y) for fft shift in frequency domain
    N, M = fp.shape
    x, y = np.meshgrid(np.arange(M), np.arange(N))
    T = (-1)**(x + y)
    fp = fp * T

    # 3. fft & filtering
    Fp = np.fft.fft2(fp)
    if 'D0' not in kwargs:
        kwargs['D0'] = min(M, N)//4
    if re.match('^B\w{2}F', fltFunc.__name__):
        if 'n' not in kwargs :
            kwargs['n'] = 2
    if re.match('^\w{1}B\w{1}F', fltFunc.__name__):
        if 'W' not in kwargs :
            kwargs['W'] = min(M, N)//8

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
    sys.path.append("../signal")
    from filters import padding_backward
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
