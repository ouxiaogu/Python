# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-06 23:50:24



Last Modified by: ouxiaogu
"""
import numpy as np
import unittest
import cv2

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
from ImFeatures import *
from ImGUI import imshowMultiple
from ImDescriptors import printImageInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../common/")
from filters import cv_gaussian_kernel, gaussian_filter, applySepFilter, fftconvolve

IMFILE = r'C:\Users\ouxiaogu\Documents\github\Canny-edge-detector\emilia.jpg'

class TestImFeatures(unittest.TestCase):
    def test_gradient(self):
        np.random.seed(0)
        im = np.random.randn(3, 3)
        G, theta = gradient(im)
        print("gradient magnitude & theta\n{}\n{}".format(G, theta))

def display():
    im = cv2.imread(IMFILE)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #flt_G = gaussian_filter(sigma=2) #cv_gaussian_kernel
    G = np.array([[2, 4,  5,  4,  2],
                   [4, 9,  12, 9,  4],
                   [5, 12, 15, 12, 5],
                   [4, 9,  12, 9,  4],
                   [2, 4,  5,  4,  2]])
    G = G.astype(np.float64)/np.sum(G)
    #print(cv_gaussian_kernel(5, 2, dtype=np.int32))
    #gim = applySepFilter(gray, flt_G, flt_G)
    gim = fftconvolve(gray, G)

    Gx, Gy = gradientXY(gim)
    G, _ = gradient(gim)
    printImageInfo(Gx)
    printImageInfo(Gy)
    printImageInfo(G)
    imshowMultiple([gim, Gx, Gy, G],
        ['gaussian', 'Gx', 'Gy', 'G'])


if __name__ == '__main__':
    # unittest.main()
    display()