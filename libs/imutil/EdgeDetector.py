# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-06 20:24:52

Edge detector

classic canny edge detector

1. Smooth input image by Gaussian Filter
2. Compute the image gradient(magnitude and angle)
3. nonmaxima suppression on gradient magnitude
4. double thresholding and trace contour by connectivity analysis

Last Modified by: ouxiaogu
"""
import numpy as np

from ImageDescriptors import gradient

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import cv_gaussian_kernel, gaussian_filter, correlate, kernelPreProc, fltGenPreProc, applySepFilter, applyKernelOperator, padding, sync_dtype

class EdgeDetector(object):
    """docstring for EdgeDetector"""
    def __init__(self, im, sigma, thresL, thresH):
        super(EdgeDetector, self).__init__()
        self.im = im
        self.sigma = sigma
        self.thresL = thresL
        self.thresH = thresH
        
    def run(self):
        # smooth
        flt_G = gaussian_filter(sigma)
        gim = applySepFilter(self.im, flt_G, flt_G)

        # gradient
        G, theta = gradient(gim)

        # nonmaxima suppression
        
