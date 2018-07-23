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

DIPPATH = r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH05'
PROJECTPATH = r'C:\Localdata\D\Book\DIP\DIPum\DIPUM2E_Projects\SAMPLE_DIPUM2E_PROJECT_IMAGES'
WORKDIR = r"C:\Localdata\D\Note\Python\misc\iCal\SEM\samples"

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
    fHist = hist_rect( normalize(power, 255, np.uint8), 100)

    imshowMultiple_TitleMatrix([gn, amp]+ [sHist, fHist], 2, 2,
        ['Gaussian Noise', 'Amplitude'], ['sptial', 'frequency'])

def try_polyroi():
    KEY_ESC = 27

    IMFILE = os.path.join(PROJECTPATH, r'FigP0501(noisy_superconductor_image).tif')
    im = cv2.imread(IMFILE, 0)

    named_window = r"draw poly roi"
    cv2.namedWindow(named_window)
    imcolor = cv2.cvtColor(im, cv2.CV_GRAY2BGR);
    cv2.setMouseCallback(name_window, collect_clicks)
    while(True):
        cv2.imshow(named_window, imcolor)
        cur_key = cv2.waitKey()
        if cur_key == KEY_ESC:
            break
        if(len(POINTS) >= 2):
            cv2.polylines(imcolor, POINTS, false, (0,255,255), 1, 8); # yellow

    cv2.destroyAllWindows()


def main():
    try_noise()

    try_polyroi()

if __name__ == '__main__':
    main()