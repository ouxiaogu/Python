# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-18 18:33:04

Unit test for spatial domain filters

Last Modified by: ouxiaogu
"""

import math
import unittest
import numpy as np

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
from SpatialFlt import *
from ImTransform import normalize
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import fftconvolve, convolve

DIPPATH = r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH03'

class TestFilters(unittest.TestCase):
    def setUp(self):
        self.shape = (5, 5)

    def test_GaussianFilter(self):
        print(GaussianFilter(self.shape) )

    def test_LaplaceFilter(self):
        print(LaplaceFilter(self.shape) )
        print(LaplaceFilter(self.shape, True) )

    def test_SobelFilter(self):
        print(SobelFilter((5,3) ) )
        print(SobelFilter((3,5), 1) )
        print(PrewittFilter((3,5), 1))

    def test_Sobel(self):
        '''
        summary the difference:
        1. python method: scipy and convolve/fftconvolve from filters.py both
           flipUD the filters, so need to use [1 0 -1], correct
        2. opencv method: no flip, directly use sum(column * kx), so
           kx = [-1 0 1], correct
        3. mxp method:  no flip, directly use sum(column * kx), but use
           kx = {1, 0, -1}, wrong
        '''
        IMFILE = os.path.join(DIPPATH, r'Fig0343(a)(skeleton_orig).tif')
        im = cv2.imread(IMFILE, 0)

        imdX_cv = cv2.Sobel(im, cv2.CV_32F, 1, 0)
        imdX_sp = convolve(im, [1, 0, -1], [1, 2, 1])

        fltShape = (3, 3)
        flt_sX = SobelFilter(fltShape)

        from scipy import signal
        bas = signal.fftconvolve(im, flt_sX, 'same')
        imdX = fftconvolve(im, flt_sX)
        np.testing.assert_almost_equal(bas, imdX, decimal=5)
        np.testing.assert_almost_equal(bas, imdX_sp, decimal=5)
        np.testing.assert_almost_equal(np.percentile(imdX, np.linspace(0, 100, 6)), np.percentile(imdX_cv, np.linspace(0, 100, 6)))
        np.testing.assert_almost_equal(bas[1:-1, 1:-1], imdX_cv[1:-1, 1:-1], decimal=5) #difference come from padding

if __name__ == '__main__':
    unittest.main()