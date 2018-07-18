# -*- coding: utf-8 -*-
"""
Created: peyang, 2018-07-16 11:26:04

DIP Image intensity transform and spatial filter pratice

Last Modified by: ouxiaogu
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../imutil")
from ImTransform import *
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
    im = cv2.medianBlur(im, 5)
    hist = hist_lines(im)

    equalizedIm = equalizeHisto(im)
    equalizedHisto = hist_lines(equalizedIm)

    imshowMultiple_TitleMatrix([im, hist, equalizedIm, equalizedHisto],
        2, 2, ['raw', 'equalized'], ['image', 'Histogram']
        , cbar=False, **KWARGS)

def try_specifyHisto():
    sfile = os.path.join(WORKDIR, r'Calaveras_v3_p3613_LDose.bmp')
    rfile = os.path.join(WORKDIR, r'Calaveras_v3_p3613_regular.bmp')

    ksize = 5
    src = cv2.imread(sfile, 0)
    src = cv2.medianBlur(src, ksize)
    shist = hist_lines(src)

    ref = cv2.imread(rfile, 0)
    ref = cv2.medianBlur(ref, ksize)
    rhist = hist_lines(ref)
    refE = equalizeHisto(ref)
    rhistE = hist_lines(refE)

    dst = specifyHisto(src, ref)
    dhist = hist_lines(dst)
    dstE = specifyHisto(src, refE)
    dhistE = hist_lines(dstE)

    imshowMultiple_TitleMatrix([src, ref, dst, refE, dstE] + [shist, rhist, dhist, rhistE, dhistE],
        2, 5, ['image', 'Histogram'], ['Low dose', 'regular', 'specified', 'reg equalized', 'specified to regular equalized'],
        cbar=False, **KWARGS)

def try_localHistoEq():
    # Elapsed time for equalizeHisto: 0.38696893308588187!
    # Elapsed time for localHistoEqualize1: 15.967883523724595!
    # Elapsed time for localHistoEqualize2: 14.640780619891302!
    # Elapsed time for localHistoEqualize3: 24.334276688725367!
    infile = os.path.join(DIPPATH, r'Fig0326(a)(embedded_square_noisy_512).tif')
    im = cv2.imread(infile, 0)
    hist = hist_lines(im)

    import timeit

    start = timeit.default_timer()
    imE = equalizeHisto(im)
    histE = hist_lines(imE)
    end = timeit.default_timer()
    print("Elapsed time for equalizeHisto: {}!".format(end - start) )

    start = timeit.default_timer()
    imLE1 = localHistoEqualize1(im)
    histLE1 = hist_lines(imLE1)
    end = timeit.default_timer()
    print("Elapsed time for localHistoEqualize1: {}!".format(end - start) )

    start = timeit.default_timer()
    imLE2 = localHistoEqualize2(im)
    histLE2 = hist_lines(imLE2)
    end = timeit.default_timer()
    print("Elapsed time for localHistoEqualize2: {}!".format(end - start) )

    start = timeit.default_timer()
    imLE3 = localHistoEqualize3(im)
    histLE3 = hist_lines(imLE3)
    end = timeit.default_timer()
    print("Elapsed time for localHistoEqualize3: {}!".format(end - start) )

    imshowMultiple_TitleMatrix([im, imE, imLE2, imLE3]+
        [hist, histE, histLE2, histLE3], 2, 4,
        ['image', 'Histogram'],
        ['raw', 'equalized', 'local equalized 2', 'local equalized 3'],
        cbar=False, **KWARGS)

def main():


    # try_equalizeHisto()

    # try_equalizeHisto_LowDose()

    # try_specifyHisto()

    # try_localHistoEq()
    # import cProfile
    # cProfile.run('try_localHistoEq()')


if __name__ == '__main__':
    main()


