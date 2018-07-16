# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-16 11:26:04

DIP Image intensity transform and spatial filter pratice

Last Modified by: ouxiaogu
"""

import os
import sys
sys.path.append("../imutil")
from ImTransform import Histogram, equalizeHisto
from ImGUI import imshowMultiple_TitleMatrix
from ImDescriptors import hist_lines
import cv2

# INPATH = r'C:\Localdata\D\Note\Python\misc\iCal\SEM\samples'
DIPPATH = r'C:\Localdata\D\Book\DIP\DIP\imagesets\DIP3E_Original_Images_CH03'
WORKDIR = r"C:\Localdata\D\Note\Python\misc\iCal\SEM\samples"
KWARGS = {'vmin': 0, 'vmax': 255}


def try_equalizeHisto():
    imnames = ['Fig0316(4)(bottom_left).tif', 'Fig0316(1)(top_left).tif', 'Fig0316(2)(2nd_from_top).tif', 'Fig0316(3)(third_from_top).tif', ]

    imfiles = [os.path.join(DIPPATH, name) for name in imnames]
    col_titles = ['dark', 'bright', 'low contrast', 'large gray range']

    images = [cv2.imread(f, 0) for f in imfiles]
    hist_rect0 = [hist_lines(im) for im in images]
    outputs = [equalizeHisto(im) for im in images]
    hist_rect1 = [hist_lines(im) for im in outputs]

    imshowMultiple_TitleMatrix(images+hist_rect0+outputs+hist_rect1,
        4, 4, ['raw', 'raw histo', 'equalizeHisto',  'equalized histo'],
        col_titles, cbar=False, **KWARGS)

def try_equalizeHisto_LowDose():
    infile = os.path.join(WORKDIR, r'calaveras_v3_LDose_p3544.bmp')

    im = cv2.imread(infile, 0)
    hist = hist_lines(im)

    equalizedIm = equalizeHisto(im)
    equalizedHisto = hist_lines(equalizedIm)

    imshowMultiple_TitleMatrix([im, hist, equalizedIm, equalizedHisto],
        2, 2, ['raw', 'equalized'], ['image', 'Histogram']
        , cbar=False, **KWARGS)

def try_specifyHisto():
    imnames = ['Fig0316(4)(bottom_left).tif', 'Fig0316(1)(top_left).tif', 'Fig0316(2)(2nd_from_top).tif', 'Fig0316(3)(third_from_top).tif', ]

    imfiles = [os.path.join(DIPPATH, name) for name in imnames]
    col_titles = ['dark', 'bright', 'low contrast', 'large gray range']

    images = [cv2.imread(f, 0) for f in imfiles]
    hist_rect0 = [hist_lines(im) for im in images]
    outputs = [equalizeHisto(im) for im in images]
    hist_rect1 = [hist_lines(im) for im in outputs]

    imshowMultiple_TitleMatrix(images+hist_rect0+outputs+hist_rect1,
        4, 4, ['raw', 'raw histo', 'equalizeHisto',  'equalized histo'],
        col_titles, cbar=False, **KWARGS)

def main():
    # try_equalizeHisto()

    try_equalizeHisto_LowDose()

if __name__ == '__main__':
    main()


