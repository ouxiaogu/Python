# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-18 18:33:04

Unit test for spatial domain filters

Last Modified by: ouxiaogu
"""

import math
import unittest
import numpy as np
import cv2

import sys
import os.path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../common")
import logger
log = logger.setup(name='SpatialTest', level='debug')

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from SpatialFlt import *
from ImTransform import normalize

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../../signal")
from filters import fftconvolve, convolve

DIPPATH = r'D:\book\DIP\DIP\imageset\DIP3E_Original_Images_CH03'
# DIPPATH = r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH03'

class TestSpFilters(unittest.TestCase):
    def setUp(self):
        self.shape = (5, 5)

    def test_GaussianFilter(self):
        log.debug('GaussianFilter :\n'+ str(GaussianFilter(3)) )
        kwargs = {'dtype': np.int32}
        log.debug('GaussianFilter, w/o normalize, int:\n'+ str(GaussianFilter(3, False, **kwargs) ))

    def test_LaplaceFilter(self):
        log.debug('LaplaceFilter :\n'+ str(LaplaceFilter(self.shape)))

    def test_SobelFilter(self):
        log.debug('SobelFilter 5x3, X:\n' + str(SobelFilter((5,3), 'x')))
        log.debug('PrewittFilter 3x5, Y:\n' + str(PrewittFilter((3,5), 'y')))

    def test_Sobel(self):
        '''
        summary the difference:
        1. python method: scipy and convolve/fftconvolve from filters.py both
           flipUD the filters, so need to use [1 0 -1], correct
        2. opencv method: no flip, directly use sum(column * kx), so
           kx = [-1 0 1], correct, another thing is cv2 by default no scale for Sobel
        3. mxp method:  no flip, directly use sum(column * kx), but use
           kx = {1, 0, -1}, wrong
        '''
        IMFILE = os.path.join(DIPPATH, r'Fig0343(a)(skeleton_orig).tif')
        im = cv2.imread(IMFILE, 0)

        imdX_cv = cv2.Sobel(im, cv2.CV_32F, 1, 0, scale=0.25)
        
        imdX_sp = convolve(im, [1, 0, -1], 0.25*np.array([1, 2, 1])) # spatial convolve

        fltShape = (3, 3)
        flt_sX = SobelFilter(fltShape, axis=1)
        log.debug("sobel x: \n{} ".format(flt_sX))

        from scipy import signal
        bas = signal.fftconvolve(im, flt_sX, 'same') #bas scipy
        imdX = fftconvolve(im, flt_sX)
        np.testing.assert_almost_equal(bas, imdX, decimal=5)
        np.testing.assert_almost_equal(bas, imdX_sp, decimal=5)
        log.debug("sobel fftconvolve &cv:\n {}\n {}".format(np.percentile(imdX, np.linspace(0, 100, 6)), np.percentile(imdX_cv, np.linspace(0, 100, 6))))
        np.testing.assert_almost_equal(bas[1:-1, 1:-1], imdX_cv[1:-1, 1:-1], decimal=5) #difference come from padding
        np.testing.assert_almost_equal(np.percentile(imdX, np.linspace(0, 100, 6)), np.percentile(imdX_cv, np.linspace(0, 100, 6)))
        


    def test_ContraHarmonicMean(self):
        mypath = os.path.join(DIPPATH, '../DIP3E_Original_Images_CH05')
        ksize = 3
        m = n = ksize//2
        r, c = 35, 88
        IMFILE = os.path.join(mypath, r'Fig0508(b)(circuit-board-salt-prob-pt1).tif')
        im = cv2.imread(IMFILE, 0)
        im = im[(r-n):(r+n+1), (c-m):(c+m+1)]
        ContraHarmonicMean(im, 3, -1.5)

    def test_TrimmedMean(self):
        np.random.seed(0)
        im = np.random.randn(3, 3)
        log.debug('im\n{}'.format(str(im)))
        log.debug('TrimmedMean(im, 3, 8)\n{}\n'.format(str(TrimmedMean(im, 3, 8))))
        np.testing.assert_almost_equal(TrimmedMean(im, 3, 0)[1,1], np.mean(im))

    def test_getMeanXYKernel(self):
        kx, ky = getMeanXYKernel(3, 'box', False)
        np.testing.assert_equal(kx, [1, 1, 1])
        np.testing.assert_equal(ky, [1, 1, 1])

        kx, ky = getMeanXYKernel(3, 'gaussian', True)
        log.debug('Gaussian kx, ky:\n'+ str(kx)+ '\n'+ str(ky))
        bas =  GaussianFilter(3, normalize=True)
        ref = np.matmul(ky.reshape((3, 1)),  kx.reshape((1, 3)))
        log.debug("getMeanXYKernel's 'gaussian' and GaussianFilter\n{}\n {}".format(ref, bas))
        np.testing.assert_almost_equal(bas, ref)

if __name__ == '__main__':
    unittest.main()