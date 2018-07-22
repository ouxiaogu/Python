# -*- coding: utf-8 -*-
"""
Created: ouxiaogu, 2018-07-22 21:30:20

image degradation and restoration 

Last Modified by: ouxiaogu
"""

import numpy as np 

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../imutil")
from ImGUI import imshowMultiple, imshowMultiple_TitleMatrix
from ImDescriptors import im_fft_amplitude_phase, hist_rect, printImageInfo, hist_lines
from ImTransform import normalize, intensityTransform

def try_noise():
    shape = (300, 300)
    gn = np.random.normal(0.5, 0.2, shape )
    printImageInfo(gn)
    amp, _ = im_fft_amplitude_phase(gn)
    printImageInfo(amp)
    araw, _ = im_fft_amplitude_phase(gn, raw_amplitude=True)
    amax = np.amax(araw)
    araw = intensityTransform(araw, lambda x: 0 if x==amax else x)
    power = araw * araw

    sHist = hist_lines( normalize(gn, 255, np.uint8))
    fHist = hist_lines( normalize(power, 255, np.uint8))

    imshowMultiple_TitleMatrix([gn, amp]+ [sHist, fHist], 2, 2,
        ['Gaussian Noise', 'Amplitude'], ['sptial', 'frequency'])

def main():
    try_noise()

if __name__ == '__main__':
    main()