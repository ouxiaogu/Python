"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-30 14:10:06

Last Modified by: ouxiaogu

MathUtil: python math utility functions
"""

import math
import numpy as np

def sigmoid(I, thres, sigmoidK):
    '''sigmoid function: smaller sigmoidK, faster transition'''
    assert(sigmoidK>0)
    res = 1./ (1 + math.exp(-(I - thres)/sigmoidK))
    return res

def rectFunc(w, slices=None, extend=1):
    '''1D rect window function: w is width, height is 1/w, symmetric'''
    assert(w > 0)
    if(slices is None):
        slices = 4
    else:
        slices = max(4, int(slices))
    if slices % 2 != 0:
        slices += 1
    extend = max(1, int(extend) )
    X = np.linspace( -(1 + extend)*w/2., (1 + extend)*w/2., (1+extend)*slices + 1)
    Y = [1./w if x <= w/2. and x >= -w/2. else 0 for x in X]
    return zip(X, Y)

def sinc(x):
    ''' `\sin(\pi x)/(\pi x)` np has sinc function'''
    X = np.asarray(x)
    Y = np.array([math.sin(math.pi * x)/(math.pi * x) if x != 0 else 1 for x in X]  )
    return Y

if __name__ == '__main__':
    '''test 1, sigmoid'''
    print("\ntest 1, sigmoid\n")
    hlfSz = 5
    I = [i/float(hlfSz) for i in range(-hlfSz, hlfSz+1)]
    thres = 0.2
    sigmoidK=0.2
    I = [sigmoid(i, thres, sigmoidK) for i in I]
    print(I)

    '''test 2, rectFunc'''
    print("\ntest 2, rectFunc\n")
    x, y = zip(*rectFunc(2, 10, 1))
    print(x)
    print(y)

    '''test 3, sinc'''
    print("\ntest 3, sinc:\n")
    x = np.linspace(-5, 5, 21)
    print(x)
    print(sinc(x) )