# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-20 10:52:30

Utility to process matrix

Last Modified by: ouxiaogu
"""

import numpy as np

import sys
import os.path
# print(sys.version_info )
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../imutil")
from SpatialFlt import *

epslmt = 1e-9

def decompose2DFilter(flt2d):
    '''
    [when 2D filter can separate into two 1D filters?](https://stackoverflow.com/questions/51223006/decompose-2d-filter-kernel-into-1d-kernels/51223950)
    '''
    if np.ndim(flt2d) != 2:
        raise ValueError("decompose2DFilter only support 2D filters!\n")
    U, S, V = np.linalg.svd(flt2d)
    if np.count_nonzero( np.abs(S) > epslmt) != 1:
        print(U, S, V, sep='\n')
        raise ValueError("decompose2DFilter, input flt2d is not separable!\n")
    return U[:, 0]*np.sqrt(S[0]), V[0, :]*np.sqrt(S[0])

def decompose2DIntegerFilter(flt2d):
    '''
    [Fast way to decompose separable integer 2D filter coefficients](https://dsp.stackexchange.com/questions/1868/fast-efficient-way-to-decompose-separable-integer-2d-filter-coefficients)
    '''
    if 'int' not in str(flt2d.dtype):
        raise ValueError('decompose2DIntegerFilter, only support array of int dtype, input array is in {} type!\n'.format(str(flt2d.dtype)))
    X = flt2d[0, :]
    Y = flt2d[:, 0]
    gcdX = gcdArray(X)
    gcdY = gcdArray(Y)
    X = X/gcdX
    Y = Y/gcdY
    return Y, X


def gcd(a, b):
    '''
    a = (a//b) * b + a%b
    (a//b) * b has all b's divisors, just need to compute gcd(b, a%b)
    '''
    if b == 0:
        return a
    else:
        return gcd(b, a%b)

def gcdArray(a):
    if a.ndim != 1 or len(a) < 2:
        raise ValueError("gcdArray, need 1D array with length > 2!\n")
    res = gcd(a[0], a[1])
    for i in range(2, len(a)):
        res = gcd(res, a[i])
    return res

def test_decompose2DFilter():
    print(decompose2DFilter(GaussianFilter((3,3) ) ), sep='\n')
    print(decompose2DFilter(SobelFilter((3,3), 1) ), sep='\n')

    a = np.array([-1.31607401,  0.        ,  1.31607401])
    b = np.array([-0.75983569, -1.51967137, -0.75983569])
    print(np.matmul(a.reshape((3,1) ),  b.reshape((1,3) ) ) )

    print(decompose2DFilter(np.matmul(LAPLACE_DIRECTIONAL.reshape((3,1) ),  LAPLACE_POSITION.reshape((1,3) ) ) ) )

def test_gcd():
    np.testing.assert_equal(10, gcd(60, 50))
    np.testing.assert_equal(1, gcd(1, 13))
    np.testing.assert_equal(2**2 * 5**3 * 7**2, gcd(2**3 * 5**3 * 7**5, 2**2 * 5**4 * 7**2))

def test_decompose2DIntegerFilter():
    # print(decompose2DIntegerFilter(GaussianFilter((3,3) ), sep='\n')
    print(decompose2DIntegerFilter(SobelFilter((3,3), 1, dtype=np.int32) ), sep='\n')
    print(getDerivKernels('Sobel', 0, 1) )
    print(decompose2DIntegerFilter(np.ones((3,3), dtype=np.int32) ) )

def main():
    # test_decompose2DFilter()

    # test_gcd()

    test_decompose2DIntegerFilter()

if __name__ == '__main__':
    main()