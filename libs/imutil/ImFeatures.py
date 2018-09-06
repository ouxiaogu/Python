# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-09-06 23:48:05



Last Modified by: ouxiaogu
"""

import numpy as np

# mutual import error if add gradient into ImageDescriptors
# from SpatialFlt import SobelFilter
from SpatialFlt import SobelFilter

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../signal")
from filters import cv_gaussian_kernel, applySepFilter, fftconvolve

def gradient(src, tan2=True):
    """
    Compute the image gradient(magnitude and angle)
    Using fft Sobel filter
    try np.arctan and arctan2
    
    Returns
    -------
    (G, theta): tuple
        magnitude and angle of gradient
    """
    tanfuc = np.arctan2 if tan2 else np.arctan
    SobelX = SobelFilter(axis='x')
    SobelY = SobelFilter(axis='y')
    Gx = fftconvolve(src, SobelX)
    Gy = fftconvolve(src, SobelY)
    G = np.sqrt(Gx**2, Gy**2)
    theta = tanfuc(Gy, Gx) * 180 / np.pi
    return (G, theta)